from pathlib import Path
from Connect import connect_to_db
from Customer_behavior import CustomerBehavior
from Association import AssociationRule


def main():
    # Get query file path
    sql_path = Path('../data/market_basket.sql').resolve()

    # make a query file to connect data base
    with open(file=str(sql_path), mode='r') as file:
        query_file = file.read()

    df = connect_to_db(str(query_file))

    association_rule = AssociationRule()
    association_result, info = association_rule.recommendation(
        dataframe=df,
        PersonID=30000001
    )

    # TODO: unpack `info` to catch subgroupname, city, and location of customer
    customer_behave = CustomerBehavior(association_result, info)
    _, _ = customer_behave.similar_user()


if __name__ == "__main__":
    main()
