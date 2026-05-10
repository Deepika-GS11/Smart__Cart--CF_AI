import pickle
import sqlite3
import pandas as pd
import numpy as np

class RecommendationEngine:
    def __init__(self):
        data = pickle.load(open('recommendation_data.pkl', 'rb'))
        self.matrix = data['matrix']
        self.similarity = data['similarity']
        self.users = data['users']
    
    def get_recommendations(self, user_id, n_recs=10):
        # Get all products from database
        conn = sqlite3.connect('products.db')
        products_df = pd.read_sql('SELECT product_id FROM products', conn)
        all_products = products_df['product_id'].tolist()
        conn.close()
        
        # New user fallback - recommend popular products
        if user_id not in self.users:
            conn = sqlite3.connect('products.db')
            recs = pd.read_sql('SELECT product_id, mean FROM products ORDER BY mean DESC LIMIT 10', conn)
            conn.close()
            return [(row['product_id'], float(row['mean'])) for _, row in recs.iterrows()]
        
        # Find similar users and make recommendations
        user_idx = self.users.index(user_id)
        recommendations = []
        
        for product in all_products:
            if product not in self.matrix.columns:
                continue
            
            # Weighted average rating from similar users
            weighted_score = 0
            total_weight = 0
            for i, sim in enumerate(self.similarity[user_idx]):
                if i != user_idx and self.matrix.iloc[i][product] > 0:
                    weighted_score += sim * self.matrix.iloc[i][product]
                    total_weight += sim
            
            if total_weight > 0:
                recommendations.append((product, weighted_score / total_weight))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recs]
