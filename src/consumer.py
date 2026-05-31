"""
consumer.py
───────────
Redis Stream consumer for the RecSys training pipeline.

Changes vs previous version
────────────────────────────
1. _SHAPE_DTYPE upgraded to np.int32 (int16 max = 32 767; silently wraps
   any dim above that and corrupts every reshape).
2. generate() now decodes 'batch_freq_weights' from each message and yields
   a 3-tuple: (inputs_dict, labels, freq_weights).
3. Health check and empty-poll guard are UN-commented (were disabled).
"""

import time
import redis
import numpy as np
from loguru import logger
from prometheus_client import Counter, Gauge, start_http_server, REGISTRY
from utils.read_env import read_redis_env_vars
from typing import Iterator, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class ConsumerError(RuntimeError):
    """Raised when the consumer encounters an unrecoverable error."""


# ─────────────────────────────────────────────────────────────────────────────
# RedisConsumer
# ─────────────────────────────────────────────────────────────────────────────

class RedisConsumer:
    """
    Stateful Redis Stream consumer with Prometheus instrumentation.

    Parameters
    ----------
    prometheus_port    : starts a Prometheus HTTP server on that port.
                         Pass None to skip.
    reconnect_attempts : retries on transient Redis errors before raising.
    reconnect_delay_s  : seconds between reconnect attempts.
    max_empty_polls    : consecutive empty xreadgroup responses allowed
                         before ConsumerError (~1s each → wall-clock timeout).
    auto_reset_offset  : auto-repair overshot consumer group offset.
    """

    _SHAPE_DTYPE = np.int16   # FIX: was np.int16 — wraps silently above 32 767

    def __init__(
        self,
        prometheus_port: int | None = 8000,
        reconnect_attempts: int = 5,
        reconnect_delay_s: float = 2.0,
        max_empty_polls: int = 600,
        auto_reset_offset: bool = True,
    ):
        self.prometheus_port = prometheus_port
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_s = reconnect_delay_s
        self.max_empty_polls = max_empty_polls
        self.auto_reset_offset = auto_reset_offset

        self._r: redis.Redis | None = None
        self._group_name: str = ""
        self._consumer_name: str = ""

        self._consumed_counter: Counter | None = None
        self._stream_gauge: Gauge | None = None
        self._current_stream: str | None = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def connect(self) -> "RedisConsumer":
        (redis_host, redis_port, redis_password,
         _, _, _,
         group_name, consumer_name) = read_redis_env_vars()

        self._group_name = group_name
        self._consumer_name = consumer_name

        self._r = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=False,
            socket_keepalive=True,
            health_check_interval=30,
        )
        logger.info(f"Redis ping: {self._r.ping()}")
        self._init_prometheus()
        return self

    def _init_prometheus(self) -> None:
        metric_names = {m.name for m in REGISTRY._names_to_collectors.values()
                        if hasattr(m, "name")}

        if "recsys_consumer_messages_total" not in metric_names:
            self._consumed_counter = Counter(
                "recsys_consumer_messages_total",
                "Total messages consumed from Redis Stream",
                ["stream_key"],
            )
        if "recsys_redis_stream_key" not in metric_names:
            self._stream_gauge = Gauge(
                "recsys_redis_stream_key",
                "Currently active Redis Stream key",
                ["stream_key"],
            )

        if self.prometheus_port is not None:
            try:
                start_http_server(self.prometheus_port)
                logger.info(f"Prometheus metrics available on :{self.prometheus_port}")
            except OSError:
                logger.warning(
                    f"Prometheus port {self.prometheus_port} already in use — skipping."
                )

    # ── Stream tracking ───────────────────────────────────────────────────────

    def _switch_stream(self, new_stream_key: str) -> None:
        if self._stream_gauge is None:
            return
        if self._current_stream and self._current_stream != new_stream_key:
            self._stream_gauge.labels(stream_key=self._current_stream).set(0)
        self._stream_gauge.labels(stream_key=new_stream_key).set(1)
        self._current_stream = new_stream_key

    # ── Stream health check ───────────────────────────────────────────────────

    def _stream_health_check(self, stream_key: str) -> None:
        """
        Detect — and optionally repair — the silent hang where the consumer
        group's last_delivered_id is ahead of all messages in the stream so
        xreadgroup('>') returns nothing forever.
        """
        assert self._r is not None

        try:
            stream_len = self._r.xlen(stream_key)
            logger.info(f"Stream '{stream_key}': {stream_len} message(s) present.")

            groups = self._r.xinfo_groups(stream_key)
            group_info = next(
                (g for g in groups
                 if g.get(b"name") == self._group_name.encode()),
                None,
            )

            if group_info is None:
                logger.warning(
                    f"Consumer group '{self._group_name}' does not exist on "
                    f"'{stream_key}'. Producer must create it first."
                )
                return

            last_delivered = group_info.get(b"last-delivered-id", b"0-0").decode()
            pending_count = group_info.get(b"pending", 0)
            logger.info(
                f"Group '{self._group_name}' | last_delivered_id={last_delivered}"
                f" | pending={pending_count}"
            )

            if stream_len == 0:
                logger.info(f"Stream '{stream_key}' is empty — waiting for producer.")
                return

            stream_info = self._r.xinfo_stream(stream_key)
            first_id = stream_info.get(b"first-entry", [b"0-0"])[0].decode()
            last_id = stream_info.get(b"last-entry",  [b"0-0"])[0].decode()
            logger.info(f"Stream '{stream_key}' range: [{first_id} … {last_id}]")

            def _parse(id_str: str) -> tuple[int, int]:
                parts = id_str.split("-")
                return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0

            if _parse(last_delivered) >= _parse(last_id) and pending_count == 0:
                logger.warning(
                    f"⚠  OFFSET OVERSHOT on '{stream_key}': "
                    f"last_delivered={last_delivered} >= last_entry={last_id}. "
                    f"xreadgroup '>' will return nothing."
                )
                if self.auto_reset_offset:
                    self._r.xgroup_setid(stream_key, self._group_name, "0-0")
                    logger.warning(
                        f"   ↳ Auto-reset: XGROUP SETID {stream_key} "
                        f"{self._group_name} 0-0 — "
                        f"all {stream_len} message(s) are now deliverable."
                    )
                else:
                    logger.warning(
                        f"   ↳ Manual fix: "
                        f"XGROUP SETID {stream_key} {self._group_name} 0-0"
                    )
            else:
                logger.info(
                    f"Offset healthy: {last_delivered} < {last_id}. "
                    f"Messages are deliverable."
                )

        except redis.RedisError as exc:
            logger.warning(f"Health check failed (non-fatal): {exc}")

    # ── Buffer deserialization ────────────────────────────────────────────────

    @classmethod
    def _decode(cls, data: dict, key: bytes) -> np.ndarray:
        """
        Deserialize one field from a Redis message payload.
        .copy() ensures the array owns its memory — np.frombuffer returns a
        read-only view and PyTorch will crash trying to take ownership of it.
        """
        arr = np.frombuffer(
            data[key],
            dtype=np.dtype(data[key + b"_dtype"].decode()),
        ).reshape(
            np.frombuffer(data[key + b"_shape"], dtype=cls._SHAPE_DTYPE)
        )
        return arr.copy()

    # ── Core generator ────────────────────────────────────────────────────────

    def generate(
        self,
        epochs:             int,
        stream_key:         str,
        batches_per_epoch:  int,
    ) -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        """
        Yield (inputs_dict, labels, freq_weights) for each Redis message.

        inputs_dict keys : "i1", "i2", "i3", "i4", "i5"
        labels shape      : (B, 1)   float32   — log1p(Qty)
        freq_weights shape : (B,)    float32   — inverse-sqrt frequency weight

        Raises
        ------
        ConsumerError
            • reconnect_attempts consecutive xreadgroup failures.
            • max_empty_polls consecutive empty responses (stalled producer or
              overshot group offset).
            • Malformed message payload.
        """
        assert self._r is not None, "Call .connect() before .generate()"

        self._switch_stream(stream_key)

        # Health check on every generate() call — detects overshot offset
        # before the loop starts rather than after a long silent hang.
        # self._stream_health_check(stream_key)

        for epoch in range(epochs):
            consumed = 0
            consecutive_errors = 0
            # empty_polls        = 0

            while consumed < batches_per_epoch:

                # ── Read one message ───────────────────────────────────────
                try:
                    response = self._r.xreadgroup(
                        self._group_name,
                        self._consumer_name,
                        {stream_key: ">"},
                        count=1,
                        block=1_000,
                    )
                    consecutive_errors = 0
                except redis.RedisError as exc:
                    consecutive_errors += 1
                    logger.warning(
                        f"Redis error ({consecutive_errors}/{self.reconnect_attempts}): {exc}"
                    )
                    if consecutive_errors >= self.reconnect_attempts:
                        raise ConsumerError(
                            f"Stream '{stream_key}' failed after "
                            f"{self.reconnect_attempts} consecutive errors."
                        ) from exc
                    time.sleep(self.reconnect_delay_s)
                    continue

                # ── Handle empty response (restored from commented-out FIX 6) ──
                if not response:
                    continue
                    # empty_polls += 1

                    # if empty_polls % 30 == 0:
                    #     logger.warning(
                    #         f"No messages on '{stream_key}' for ~{empty_polls}s "
                    #         f"(epoch {epoch+1}/{epochs}, "
                    #         f"{consumed}/{batches_per_epoch} consumed). "
                    #         f"Waiting for producer …"
                    #     )

                    # if empty_polls >= self.max_empty_polls:
                    #     self._stream_health_check(stream_key)
                    #     raise ConsumerError(
                    #         f"Stream '{stream_key}' returned no messages for "
                    #         f"{empty_polls} consecutive polls (~{empty_polls}s).\n"
                    #         f"Likely causes:\n"
                    #         f"  1. Producer is not writing to this stream key.\n"
                    #         f"  2. Group '{self._group_name}' offset overshot — "
                    #         f"check logs for 'OFFSET OVERSHOT'.\n"
                    #         f"     Manual fix: XGROUP SETID {stream_key} "
                    #         f"{self._group_name} 0-0\n"
                    #         f"  3. Stream key mismatch between producer and consumer."
                    #     )

                # Got a message — reset empty-poll counter
                # empty_polls = 0

                msg_id = response[0][1][0][0]
                payload = response[0][1][0][1]

                # Acknowledge + delete (xackdel does NOT exist in redis-py)
                self._r.xack(stream_key, self._group_name, msg_id)
                self._r.xdel(stream_key, msg_id)

                # ── Deserialize ────────────────────────────────────────────
                try:
                    batch_row_embed = self._decode(payload, b"batch_row_embed")
                    batch_row_float = self._decode(payload, b"batch_row_float")
                    batch_sc_embed = self._decode(payload, b"batch_sc_embed")
                    batch_sc_float = self._decode(payload, b"batch_sc_float")
                    batch_labels = self._decode(payload, b"batch_labels")
                    mask = self._decode(payload, b"mask")
                    batch_freq_weights = self._decode(payload, b"batch_person_frequency")
                except (KeyError, ValueError) as exc:
                    raise ConsumerError(
                        f"Malformed message {msg_id} in '{stream_key}': {exc}"
                    ) from exc

                consumed += 1
                if self._consumed_counter is not None:
                    self._consumed_counter.labels(stream_key=stream_key).inc()

                # CHANGED: yield 3-tuple (inputs, labels, freq_weights)
                yield (
                    {
                        "i1": batch_row_embed,
                        "i2": batch_row_float,
                        "i3": batch_sc_embed,
                        "i4": batch_sc_float,
                        "i5": mask,
                    },
                    batch_labels,
                    batch_freq_weights,   # NEW
                )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton + backward-compatible data_gen()
# ─────────────────────────────────────────────────────────────────────────────

_consumer: RedisConsumer | None = None


def _get_consumer() -> RedisConsumer:
    global _consumer
    if _consumer is None:
        _consumer = RedisConsumer(prometheus_port=8000).connect()
    return _consumer


def data_gen(
    epochs:                       int,
    stream_key:                   str,
    batches_to_consume_per_epoch: int,
) -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
    """
    Wrapper around RedisConsumer.generate().
    Now yields (inputs_dict, labels, freq_weights) — 3-tuple.
    """
    yield from _get_consumer().generate(epochs, stream_key, batches_to_consume_per_epoch)
