import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

print("🚀 Starting Data Pipeline...")

# Extract
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print("✅ Raw Data Loaded!")
print(df.head())

# Preprocessing
df.loc[0:5, 'sepal length (cm)'] = None  
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

df.fillna(df.mean(), inplace=True)
print("\n✅ Missing Values Handled!")
print(df.isnull().sum())

# Transformation
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.iloc[:, :-1])
df_scaled = pd.DataFrame(scaled_data, columns=iris.feature_names)
df_scaled['target'] = df['target']
print("\n✅ Data Transformation Complete!")
print(df_scaled.head())

# Load
df_scaled.to_csv('processed_iris.csv', index=False)
print("\n✅ Data Pipeline Complete! Processed data saved as 'processed_iris.csv''')

Output:
🚀 Starting Data Pipeline...
✅ Raw Data Loaded!
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                5.1               3.5                1.4               0.2       0
1                4.9               3.0                1.4               0.2       0
2                4.7               3.2                1.3               0.2       0
3                4.6               3.1                1.5               0.2       0
4                5.0               3.6                1.4               0.2       0

Missing Values Before Cleaning:
sepal length (cm)    6
sepal width (cm)     0
petal length (cm)    0
petal width (cm)     0
target               0
dtype: int64

✅ Missing Values Handled!
sepal length (cm)    0
sepal width (cm)     0
petal length (cm)    0
petal width (cm)     0
target               0
dtype: int64

✅ Data Transformation Complete!
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0          -0.875534          1.032057          -1.340227         -1.315444       0
1          -1.049004         -0.124958          -1.340227         -1.315444       0
2          -1.222473          0.337848          -1.397064         -1.315444       0
3          -1.309208          0.106445          -1.283390         -1.315444       0
4          -0.961269          1.263460          -1.340227         -1.315444       0

✅ Data Pipeline Complete! Processed data saved as 'processed_iris.csv'