import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('data/kc_house_data_NaN.csv')

df.drop('Unnamed: 0', axis=1, inplace=True)

df['bedrooms'].replace(np.nan, df['bedrooms'].mean().round(0), inplace=True)
df['bathrooms'].replace(np.nan, df['bathrooms'].mean().round(0), inplace=True)
df['bathrooms'] = df['bathrooms'].round(0)

df['floors'] = df['floors'].round(0)

x_data = df.drop(['price', 'id', 'sqft_living15', 'sqft_lot15', 'zipcode', 'sqft_above', 'date'], axis=1)
y_data = df['price']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

def save_to_pickle(data, filename):
    file_path = f'data/{filename}'
    
    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"File '{filename}' already exists and will be replaced.")
    
    # Save the data to the file (overwrite if it exists)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"File '{filename}' saved successfully.")

save_to_pickle(x_train, 'x_train.pkl')
save_to_pickle(x_test, 'x_test.pkl')
save_to_pickle(y_train, 'y_train.pkl')
save_to_pickle(y_test, 'y_test.pkl')

model = RandomForestRegressor(bootstrap=True, max_depth = 20, n_estimators = 200)
model.fit(x_train, y_train)

model_dir = 'model'
os.makedirs(model_dir, exist_ok=True) 
model_filepath = os.path.join(model_dir, 'rfreg.pkl')  # Correctly set the file path
with open(model_filepath, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_filepath}")