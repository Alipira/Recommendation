"""
consumer.py
───────────
Redis Stream consumer for the RecSys training pipeline.

Reads pre-batched numpy arrays from a Redis Stream group, deserializes them,
and yields (inputs_dict, labels) pairs consumed by train.py's
RedisStreamDataset → DataLoader pipeline.

Fixes vs original
─────────────────
1. xackdel → xack + xdel  (xackdel does not exist in redis-py; was a silent
                             AttributeError on the first batch)
2. np.int16 for shape     → np.int32  (int16 max = 32 767; any dim above that
   deserialization            wraps silently and corrupts every reshape)
3. Module-level side       → encapsulated in RedisConsumer class; nothing runs
   effects (Redis connect,    on import, making the module testable and safe
   Prometheus server,         for multi-process DataLoader workers
   global state)
4. np.frombuffer read-only → .copy() on every buffer so PyTorch can take
   arrays                    ownership without warnings or hidden copies
5. Silent exception        → ConsumerError is re-raised so train.py's
   swallowing                 run_epoch fails loudly instead of returning
                              partial metrics silently
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

    Nothing connects or starts a server until `connect()` is called, making
    the class safe to instantiate in any context (tests, workers, etc.).

    Parameters
    ----------
    prometheus_port : int or None
        If given, starts a Prometheus HTTP server on that port.
        Pass None to skip (e.g. in tests or when metrics are not needed).
    reconnect_attempts : int
        How many times to retry xreadgroup on a transient Redis error
        before raising ConsumerError.
    reconnect_delay_s : float
        Seconds to wait between reconnect attempts.
    """
    # FIXME: change the shape to np.int16 after test if nothing happen
    _SHAPE_DTYPE = np.int16

    def __init__(
        self,
        prometheus_port: int | None = 8000,
        reconnect_attempts: int = 5,
        reconnect_delay_s: float = 2.0,
    ):
        self.prometheus_port     = prometheus_port
        self.reconnect_attempts  = reconnect_attempts
        self.reconnect_delay_s   = reconnect_delay_s

        # Populated by connect()
        self._r: redis.Redis | None = None
        self._group_name:    str    = ""
        self._consumer_name: str    = ""

        # Prometheus metrics (created lazily to avoid double-registration
        # in multi-process environments)
        self._consumed_counter: Counter | None = None
        self._stream_gauge:     Gauge   | None = None
        self._current_stream:   str | None     = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def connect(self) -> "RedisConsumer":
        """
        Open the Redis connection and start Prometheus (if configured).
        Call once before using the consumer.
        """
        (redis_host, redis_port, redis_password,
         _, _, _,
         group_name, consumer_name) = read_redis_env_vars()

        self._group_name    = group_name
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
        return self     # allow chaining: consumer = RedisConsumer().connect()

    def _init_prometheus(self) -> None:
        # Guard against double-registration when module is reloaded
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

    # ── Buffer deserialization ────────────────────────────────────────────────

    @classmethod
    def _decode(cls, data: dict, key: bytes) -> np.ndarray:
        """
        Deserialize one field from a Redis message payload.

        .copy() is called so the returned array owns its memory — required
        because np.frombuffer returns a read-only view of the raw bytes, and
        PyTorch will crash or warn when trying to take ownership of it.
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
        epochs: int,
        stream_key: str,
        batches_per_epoch: int,
    ) -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray]]:
        """
        Yield (inputs_dict, labels) for each message consumed from the stream.

        inputs_dict keys: "i1", "i2", "i3", "i4", "i5"
        labels shape:     (B, 1)  float32

        Raises
        ------
        ConsumerError
            If `reconnect_attempts` consecutive xreadgroup calls fail, or if
            an unexpected non-Redis error occurs.
        """
        assert self._r is not None, "Call .connect() before .generate()"

        self._switch_stream(stream_key)

        for epoch in range(epochs):
            logger.info(f"Consumer: starting epoch {epoch + 1}/{epochs} "
                        f"({batches_per_epoch} batches from '{stream_key}')")
            consumed = 0
            consecutive_errors = 0

            while consumed < batches_per_epoch:
                try:
                    response = self._r.xreadgroup(
                        self._group_name,
                        self._consumer_name,
                        {stream_key: ">"},
                        count=1,
                        block=1_000,
                    )
                    consecutive_errors = 0      # reset on success
                except redis.RedisError as exc:
                    consecutive_errors += 1
                    logger.warning(
                        f"Redis error ({consecutive_errors}/{self.reconnect_attempts}): {exc}"
                    )
                    if consecutive_errors >= self.reconnect_attempts:
                        raise ConsumerError(
                            f"Redis stream '{stream_key}' failed after "
                            f"{self.reconnect_attempts} consecutive errors."
                        ) from exc
                    time.sleep(self.reconnect_delay_s)
                    continue

                # Empty response means the stream has no new messages yet;
                # block=1_000 already waited 1 s, so just loop.
                if not response:
                    continue

                msg_id   = response[0][1][0][0]
                payload  = response[0][1][0][1]

                # ── Acknowledge + delete atomically ───────────────────────
                # xackdel does NOT exist in redis-py — must be two calls.
                # xack marks the message as processed in the consumer group.
                # xdel removes it from the stream to prevent memory growth.
                self._r.xack(stream_key, self._group_name, msg_id)
                self._r.xdel(stream_key, msg_id)

                # ── Deserialize ────────────────────────────────────────────
                try:
                    batch_row_embed = self._decode(payload, b"batch_row_embed")  # (B, 17) int
                    batch_row_float = self._decode(payload, b"batch_row_float")  # (B, 91) float32
                    batch_sc_embed  = self._decode(payload, b"batch_sc_embed")   # (B, T, 5) int
                    batch_sc_float  = self._decode(payload, b"batch_sc_float")   # (B, T, 5) float32
                    batch_labels    = self._decode(payload, b"batch_labels")     # (B, 1) float32
                    mask            = self._decode(payload, b"mask")             # (B, T, 1) float32
                except (KeyError, ValueError) as exc:
                    raise ConsumerError(
                        f"Malformed message {msg_id} in stream '{stream_key}': {exc}"
                    ) from exc

                consumed += 1
                if self._consumed_counter is not None:
                    self._consumed_counter.labels(stream_key=stream_key).inc()

                yield (
                    {
                        "i1": batch_row_embed,
                        "i2": batch_row_float,
                        "i3": batch_sc_embed,
                        "i4": batch_sc_float,
                        "i5": mask,
                    },
                    batch_labels,
                )

            logger.info(
                f"Consumer: epoch {epoch + 1}/{epochs} complete "
                f"({consumed} batches consumed)"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton + backward-compatible data_gen()
# ─────────────────────────────────────────────────────────────────────────────

# Lazily initialised — nothing happens at import time.
_consumer: RedisConsumer | None = None


def _get_consumer() -> RedisConsumer:
    """Return the module-level consumer, connecting on first call."""
    global _consumer
    if _consumer is None:
        _consumer = RedisConsumer(prometheus_port=8000).connect()
    return _consumer


def data_gen(
    epochs: int,
    stream_key: str,
    batches_to_consume_per_epoch: int,
) -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray]]:
    """
    Backward-compatible wrapper around RedisConsumer.generate().

    This is the function imported by train.py — signature is unchanged.
    """
    yield from _get_consumer().generate(epochs, stream_key, batches_to_consume_per_epoch)