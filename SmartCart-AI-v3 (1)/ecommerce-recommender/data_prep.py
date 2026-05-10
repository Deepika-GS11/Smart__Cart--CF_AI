import pandas as pd
import numpy as np
import sqlite3
import pickle
from sklearn.metrics.pairwise import cosine_similarity

print("🚀 E-Commerce Recommendation System - REAL PRODUCT NAMES")

# REAL E-COMMERCE PRODUCTS (50 realistic names)
products = [
    "iPhone 15 Pro", "Samsung Galaxy S25", "MacBook Air M3", "Dell XPS 13", "Sony WH-1000XM5",
    "AirPods Pro 2", "Apple Watch Ultra", "Nike Air Force 1", "Adidas Ultraboost", "Ray-Ban Wayfarer",
    "Canon EOS R5", "Dyson V15 Vacuum", "Kindle Paperwhite", "Fitbit Charge 6", "Bose QuietComfort",
    "Levis 501 Jeans", "North Face Jacket", "Patagonia Fleece", "Yeti Rambler Tumbler", "Instant Pot Duo",
    "Ninja Air Fryer", "Samsung 4K TV", "LG OLED TV", "PS5 Console", "Xbox Series X",
    "Oculus Quest 3", "DJI Mini Drone", "GoPro Hero 12", "Apple iPad Pro", "Surface Pro 9",
    "HP Spectre x360", "Logitech MX Master", "Apple Magic Keyboard", "Razer DeathAdder", "Corsair K95 Keyboard",
    "Samsung Odyssey Monitor", "BenQ Eye-Care Monitor", "Echo Dot 5th Gen", "Google Nest Hub", "Ring Video Doorbell",
    "Arlo Pro 4 Camera", "Philips Hue Bulbs", "Roborock S8 Robot Vacuum", "Eufy RoboVac", "Breville Coffee Maker"
]

# Limit to 50 products for demo
products = products[:50]

# Generate ratings data
np.random.seed(42)
users = range(1, 51)
ratings = []
for user in users:
    n_ratings = np.random.randint(15, 35)
    user_products = np.random.choice(products, n_ratings, replace=False)
    for product in user_products:
        rating = np.random.randint(1, 6)
        ratings.append([user, product, rating])

df = pd.DataFrame(ratings, columns=['user_id', 'product_id', 'rating'])
df.to_csv('ratings.csv', index=False)
print(f"✅ Dataset: {len(df)} ratings with REAL products!")

# SQLite database
conn = sqlite3.connect('products.db')
product_stats = df.groupby('product_id')['rating'].agg(['count', 'mean']).round(2)
product_stats.to_sql('products', conn, if_exists='replace')
conn.close()

# User-item matrix + similarity
user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)

data = {'matrix': user_item_matrix, 'similarity': user_similarity, 'users': list(user_item_matrix.index)}
with open('recommendation_data.pkl', 'wb') as f:
    pickle.dump(data, f)

print("✅ REAL PRODUCTS READY! Restart app: Ctrl+C then python app.py")
