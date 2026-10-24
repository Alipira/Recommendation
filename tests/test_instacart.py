import pandas as pd
import pytest

from hybridrec.data.instacart import load_instacart


@pytest.fixture()
def instacart_dir(tmp_path):
    pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "user_id": [7, 7, 7, 8, 8],
        "eval_set": ["prior", "prior", "train", "prior", "test"],
        "order_number": [1, 2, 3, 1, 2],
        "order_dow": 0, "order_hour_of_day": 10,
        "days_since_prior_order": [None, 5.0, 0.0, None, 3.0],
    }).to_csv(tmp_path / "orders.csv", index=False)
    pd.DataFrame({"order_id": [1, 1, 2, 4], "product_id": [10, 11, 10, 12],
                  "add_to_cart_order": 1, "reordered": 0}).to_csv(tmp_path / "order_products__prior.csv", index=False)
    pd.DataFrame({"order_id": [3], "product_id": [11], "add_to_cart_order": 1,
                  "reordered": 1}).to_csv(tmp_path / "order_products__train.csv", index=False)
    pd.DataFrame({"product_id": [10, 11, 12], "product_name": ["Milk", "Bread", "Tea"],
                  "aisle_id": [1, 2, 2], "department_id": [1, 1, 2]}).to_csv(tmp_path / "products.csv", index=False)
    pd.DataFrame({"aisle_id": [1, 2], "aisle": ["dairy", "bakery"]}).to_csv(tmp_path / "aisles.csv", index=False)
    pd.DataFrame({"department_id": [1, 2], "department": ["fresh", "drinks"]}).to_csv(
        tmp_path / "departments.csv", index=False)
    return tmp_path


def test_load_instacart(instacart_dir):
    ds = load_instacart(instacart_dir)
    inter = ds.interactions
    assert set(inter["basket_id"]) == {"1", "2", "3", "4"}          # test-set order 5 dropped
    ts = inter.drop_duplicates("basket_id").set_index("basket_id")["ts"]
    assert ts["1"] < ts["2"] < ts["3"]                               # time rebuilt from gaps
    assert ts["3"] - ts["2"] < 0.01                                  # same-day order kept after
    assert ds.item_text()["10"] == "Milk (dairy, fresh)"


def test_load_instacart_sample_users(instacart_dir):
    ds = load_instacart(instacart_dir, max_users=1, seed=0)
    assert ds.interactions["user_id"].nunique() == 1
