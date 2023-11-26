import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


class CustomerBehavior:
    def __init__(self, dataframe, customer_subgroup, user_id, association):
        self.df = dataframe
        self.user_id = user_id
        self.association = association
        if isinstance(customer_subgroup, str):
            self.customer_group = customer_subgroup

    def calculate_cosine_similarity(self, ):
        # Filter data for the specified customer subgroup
        df_customer_group = self.df.query('CustomerSubGroupName == @self.customer_group')

        # Create a purchase matrix where rows are users, columns are items,
        # and values are purchase counts
        purchase_matrix = df_customer_group.groupby(['PersonID', 'Group4Name'])['Qty']\
            .sum().unstack().reset_index().fillna(0).set_index('PersonID')

        # Calculate cosine similarity between users
        user_similarity = cosine_similarity(purchase_matrix.to_numpy())

        return user_similarity, purchase_matrix

    def similar_user(self, similarity_threshold=0.8):
        # Get user purchases and similarity scores
        user_similarity_matrix, purchase_matrix = self.calculate_cosine_similarity()

        user_purchases = purchase_matrix[self.user_id]
        similarity_scores = user_similarity_matrix[self.user_id]

        # Filter items based on the similarity threshold
        similar_users = np.where(user_similarity_matrix[0] >= similarity_threshold)

        return similar_users
