import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler


# Generate a reproducible 100-row dataset with missing values and one outlier.
# Result: a pandas DataFrame containing numeric features, a categorical feature, and a target.
def generate_dummy_data():
    np.random.seed(42)
    data = {
        'numeric_feature_1': np.random.normal(50, 10, 100),
        'numeric_feature_2': np.random.normal(100, 20, 100),
        'categorical_feature': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    }
    df = pd.DataFrame(data)

    df.loc[10:15, 'numeric_feature_1'] = np.nan
    df.loc[20:25, 'categorical_feature'] = np.nan
    df.loc[99, 'numeric_feature_2'] = 1000
    return df


# Fill missing categorical values with the mode and numeric values with the mean.
# Result: the same DataFrame with no missing values.
def handle_missing_values(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df


# Remove rows whose numeric values have an absolute z-score greater than the threshold.
# Result: a DataFrame containing only rows that are not statistical outliers.
def remove_outliers(df, threshold=3):
    numeric_cols = df.select_dtypes(include=[np.number])
    z_scores = np.abs(stats.zscore(numeric_cols))
    return df[(z_scores < threshold).all(axis=1)]


# Standardize each numeric feature except the target using mean 0 and standard deviation 1.
# Result: the DataFrame with scaled feature values and an unchanged target column.
def scale_features(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features_to_scale = [col for col in numeric_cols if col != 'target']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return df


# Convert categorical columns into numeric binary columns using one-hot encoding.
# Result: a fully numeric DataFrame without the first category of each categorical column.
def encode_categorical(df):
    return pd.get_dummies(df, drop_first=True)


if __name__ == "__main__":
    df = generate_dummy_data()
    df = handle_missing_values(df)
    df = remove_outliers(df)
    df = scale_features(df)
    final_df = encode_categorical(df)
    final_df.to_csv('preprocessed_dummy_data.csv', index=False)
    print("Pipeline complete! Clean data saved.")