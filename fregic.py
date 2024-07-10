import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

file_path = 'Seoul.csv'
encodings = ['euc-kr', 'utf-8', 'latin1']

for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f"Successfully read the file with encoding: {enc}")
        break
    except UnicodeDecodeError as e:
        print(f"Failed to read the file with encoding: {enc}, Error: {e}")

print("\n뚤림:")
print(df.isnull().sum())

original_df = df.copy()

def predict_missing_values(df, target_column):
    missing_rows = df[df[target_column].isnull()]
    non_missing_rows = df.dropna(subset=[target_column])
    
    feature_columns = df.columns.drop([target_column, 'date'])
    
    non_missing_rows = non_missing_rows.dropna(subset=feature_columns)
    
    if non_missing_rows.empty or missing_rows.empty:
        print(f"No data {target_column}")
        return df
    
    X_train = non_missing_rows[feature_columns].astype(float)
    y_train = non_missing_rows[target_column].astype(float)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    X_missing = missing_rows[feature_columns].astype(float)
    predicted_values = model.predict(X_missing)
    
    print(f"\n예측 {target_column}: {predicted_values}")
    
    df.loc[missing_rows.index, target_column] = predicted_values
    return df

columns_with_missing_values = df.columns[df.isnull().any()]

for column in columns_with_missing_values:
    df = predict_missing_values(df, column)

filled_rows = df[original_df.isnull().any(axis=1)]

print("\n채운 후 빈칸:")
print(df.isnull().sum())

print("\n채워진 줄:")
print(filled_rows)
