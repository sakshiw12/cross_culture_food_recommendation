# train_model.py

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your dataset (example: recipes.csv)
df = pd.read_csv('data.csv')

# Combine all text features into one for vectorization
df['combined'] = df['name'] + " " + df['ingredients'] + " " + df['cuisine'] + " " + df['diet']

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])

# Save the model components in a dictionary
model = {
    'data': df,
    'tfidf': tfidf,
    'tfidf_matrix': tfidf_matrix
}

# Save as pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved to model.pkl")




