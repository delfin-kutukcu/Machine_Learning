# Spotify Song Popularity Analysis with Machine Learning

## Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import normaltest, boxcox

## Load dataset
spotify_df = pd.read_csv('datasets/spotify.csv')

## Dataset Overview
print("Dataset Overview:\n", spotify_df.head())

## Check and visualize missing values
missing_values = spotify_df.isnull().sum()
print("\nMissing Values (%):\n", (missing_values / len(spotify_df)) * 100)

## Encode categorical variables
le = LabelEncoder()
spotify_df['track_genre'] = le.fit_transform(spotify_df['track_genre'])

## Popularity Distribution Visualization
plt.figure(figsize=(8,5))
sns.histplot(spotify_df['popularity'], kde=True)
plt.title('Popularity Distribution')
plt.savefig('visuals/popularity_distribution.png')
plt.show()

## Correlation Matrix
plt.figure(figsize=(12,10))
sns.heatmap(spotify_df.corr(), annot=True, cmap='coolwarm')
plt.title('Spotify Songs Correlation Matrix')
plt.savefig('visuals/correlation_matrix.png')
plt.show()

## Simple Linear Regression (danceability -> popularity)
X = spotify_df[['danceability']]
y = spotify_df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

simple_reg = LinearRegression().fit(X_train, y_train)
y_pred = simple_reg.predict(X_test)
print("\nSimple Linear Regression R²:", r2_score(y_test, y_pred))
print("Simple Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

## Actual vs Predicted Values Visualization
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted', alpha=0.5)
plt.xlabel('Danceability')
plt.ylabel('Popularity')
plt.title('Simple Linear Regression - Spotify Popularity')
plt.legend()
plt.savefig('visuals/simple_regression.png')
plt.show()

## Multiple Linear Regression
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
            'instrumentalness', 'liveness', 'valence', 'tempo']
X_multi = spotify_df[features]
y_multi = spotify_df['popularity']
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

multi_reg = LinearRegression().fit(X_train_m, y_train_m)
y_pred_multi = multi_reg.predict(X_test_m)
print("\nMultiple Linear Regression R²:", r2_score(y_test_m, y_pred_multi))
print("Multiple Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test_m, y_pred_multi)))

## Residual Analysis
residuals = y_test_m - y_pred_multi
plt.figure(figsize=(8,5))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution - Multiple Linear Regression')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.savefig('visuals/residuals_distribution.png')
plt.show()

## Normality Test (Original)
stat, p_val = normaltest(spotify_df['popularity'])
print("\nNormality test p-value (Original):", p_val)

## Data Transformations and Visualization
transformations = {
    'Log': np.log1p(spotify_df['popularity']),
    'Square Root': np.sqrt(spotify_df['popularity']),
    'Boxcox': boxcox(spotify_df['popularity'] + 1)[0]
}

plt.figure(figsize=(15,4))
for i, (method, data) in enumerate(transformations.items(), 1):
    plt.subplot(1, 3, i)
    sns.histplot(data, kde=True)
    plt.title(f'{method} Transformation')
plt.tight_layout()
plt.savefig('visuals/transformations.png')
plt.show()

## Normality Test after Transformations
print("\nNormality Test after Transformations:")
for method, data in transformations.items():
    stat, p = normaltest(data)
    status = 'Successful' if p > 0.05 else 'Unsuccessful'
    print(f"{method}: p-value={p:.4f} ({status})")
