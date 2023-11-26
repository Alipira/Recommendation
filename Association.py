import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules
from typing import List


class AssociationRule():
    """
    Create a pandas dataframe illustrates realationships between
    products for each CustomerSubgroup by using association rule.
    The main metrics used in this method are 'lift', 'Confidence',
    and 'Support'
    """
    def __init__(self, ):
        pass

    def encoder(self, x):
        if x <= 0:
            return 0
        return 1

    def perform_association_rule_analysis(
            self, dataframe, selected_items, shortcut_code=None, metric=None
    ):
        if metric is None:
            metric = 'lift'

        city, location, customer = selected_items[0][1], selected_items[1][1], selected_items[2][1]

        # Make a dataframe for a specific subgroup
        df_ = dataframe.query('CityName == @city and ZoneName == @location and CustomerSubGroupName == @customer')\
            .groupby(['BusinessDocNo', 'Group4Name'])['Qty'].sum().unstack().reset_index().fillna(0).set_index('BusinessDocNo')

        df_ = df_.applymap(self.encoder)

        frequent_items = apriori(df_, min_support=0.05,
                                 use_colnames=True,
                                 verbose=1,
                                 max_len=2)

        result = association_rules(frequent_items,
                                   metric='lift',
                                   min_threshold=1)

        rules = result[(result['lift'] >= 2) & (result['confidence'] >= 0.4)].\
            sort_values(by='confidence', ascending=False)
        return rules, (city, location, customer)

    def recommendation(self, dataframe, PersonID=None, shortcut_code=None):
        if PersonID is None:
            raise FileNotFoundError("PersonID is not catched!!!")

        result = dataframe.query('PersonID == @PersonID')\
            .get(['ZoneName', 'CustomerSubGroupName', 'CityName']).head(1)

        city_name = result['CityName'].values[0] if not result.empty else None
        location_name = result['ZoneName'].values[0]\
            if not result.empty else None

        Customer_type = result['CustomerSubGroupName'].values[0]\
            if not result.empty else None

        unique_cities = pd.unique(dataframe['CityName'])
        unique_locations = pd.unique(dataframe['ZoneName'])
        unique_customers = pd.unique(dataframe['CustomerSubGroupName'])

        selected_items = []

        if city_name in unique_cities:
            selected_items.append(('City', city_name))
        if location_name in unique_locations:
            selected_items.append(('Location', location_name))
        if Customer_type in unique_customers:
            selected_items.append(('Customer', Customer_type))

        if not selected_items:
            raise FileNotFoundError("City, location, or customer not found in the dataframe")

        # Perform association rule analysis on the selected items
        association_rules_result, info = self.perform_association_rule_analysis(dataframe, selected_items, shortcut_code, metric=None)
        association_rules_result.astype(
            {
                'antecedents': 'str',
                'consequents': 'str'
            }
        )
        association_rules_result.antecedents.apply(lambda x: x.strip('''frozenset(){}\''''))
        association_rules_result.consequents.apply(lambda x: x.strip('''frozenset(){}\''''))

        return association_rules_result, info
