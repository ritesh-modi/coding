import pandas as pd
import numpy as np

# Basic info about datatypes and missing values
df.info()

# Statistical summary for numeric columns
df.describe()

# Statistical summary for all columns (including categorical)
df.describe(include='all')

# Get column datatypes
df.dtypes

# Count of missing values per column
df.isnull().sum()

# Percentage of missing values
(df.isnull().sum() / len(df) * 100).round(2)

# More detailed missing value report
missing_report = pd.DataFrame({
    'missing_count': df.isnull().sum(),
    'missing_pct': (df.isnull().sum() / len(df) * 100).round(2),
    'dtype': df.dtypes
})
print(missing_report[missing_report['missing_count'] > 0])

# For NUMERIC columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].nunique()}")
    print(f"  Min: {df[col].min()}, Max: {df[col].max()}")
    print(f"  Zeros: {(df[col] == 0).sum()}")
    print(f"  Negatives: {(df[col] < 0).sum()}")
    print(f"  Outliers (>3 std): {(np.abs((df[col] - df[col].mean()) / df[col].std()) > 3).sum()}")

# For CATEGORICAL columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].nunique()}")
    print(f"  Top 5 values:\n{df[col].value_counts().head()}")
    print(f"  Empty strings: {(df[col] == '').sum()}")
    print(f"  Whitespace only: {df[col].str.isspace().sum()}")


# Check for duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# Check for columns with single value (no variance)
single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
print(f"Single-value columns: {single_value_cols}")

# Check for high cardinality in categorical columns
for col in categorical_cols:
    unique_ratio = df[col].nunique() / len(df)
    if unique_ratio > 0.95:
        print(f"{col} has very high cardinality: {unique_ratio:.2%}")

# Check for potential ID columns (all unique)
potential_ids = [col for col in df.columns if df[col].nunique() == len(df)]
print(f"Potential ID columns: {potential_ids}")


def data_quality_report(df):
    report = []
    
    for col in df.columns:
        col_info = {
            'column': col,
            'dtype': df[col].dtype,
            'missing_count': df[col].isnull().sum(),
            'missing_pct': (df[col].isnull().sum() / len(df) * 100).round(2),
            'unique_count': df[col].nunique(),
            'unique_pct': (df[col].nunique() / len(df) * 100).round(2)
        }
        
        if np.issubdtype(df[col].dtype, np.number):
            col_info.update({
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'zeros': (df[col] == 0).sum(),
                'negatives': (df[col] < 0).sum()
            })
        else:
            col_info['most_common'] = df[col].mode()[0] if len(df[col].mode()) > 0 else None
            
        report.append(col_info)
    
    return pd.DataFrame(report)

quality_df = data_quality_report(df)
print(quality_df)