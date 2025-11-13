"""
Document structure for Diplomski:
- Introductions
- Merging the datasets and preprocessing
- Exploratory Data Analysis
- Training the models
- Results and Conclusion
"""

import pandas as pd

def merge_datasets(d1, d2, d3=None):
    """Merge the provided datasets after preprocessing them. d3 is optional."""
    dfs = []
    for d in (d1, d2, d3):
        if d is None:
            continue
        dfs.append(preprocess_dataset(d))
    if not dfs:
        return pd.DataFrame()
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates().reset_index(drop=True)
    return merged

def preprocess_dataset(df):
    """Standardize column names and map YES/NO and 2/1 to 1/0. Returns cleaned DataFrame."""
    df = df.copy()
    # normalize column names
    orig_cols = list(df.columns)
    clean = {c: c.strip() for c in orig_cols}
    df.rename(columns=clean, inplace=True)

    # helper to create a normalized key
    def keyize(c):
        return c.strip().lower().replace('-', '_').replace(' ', '_')

    # map many possible column name variants to standard names
    col_map = {}
    for c in df.columns:
        k = keyize(c)
        if k in ('gender','sex'):
            col_map[c] = 'gender'
        elif 'age' == k or k.startswith('age'):
            col_map[c] = 'age'
        elif 'smok' in k:
            col_map[c] = 'smoking'
        elif 'yellow' in k and 'finger' in k:
            col_map[c] = 'yellow_fingers'
        elif 'anxiety' in k:
            col_map[c] = 'anxiety'
        elif 'peer' in k and 'press' in k:
            col_map[c] = 'peer_pressure'
        elif 'chronic' in k:
            col_map[c] = 'chronic_disease'
        elif 'fatig' in k:
            col_map[c] = 'fatigue'
        elif 'allerg' in k:
            col_map[c] = 'allergy'
        elif 'wheez' in k:
            col_map[c] = 'wheezing'
        elif 'alco' in k:
            col_map[c] = 'alcohol'
        elif 'cough' in k:
            col_map[c] = 'coughing'
        elif 'short' in k and 'breath' in k:
            col_map[c] = 'shortness_of_breath'
        elif 'swallow' in k:
            col_map[c] = 'swallowing_difficulty'
        elif 'chest' in k and 'pain' in k:
            col_map[c] = 'chest_pain'
        elif 'lung' in k and 'cancer' in k:
            col_map[c] = 'lung_cancer'
        else:
            # keep original (but keyized) if nothing matched
            col_map[c] = k

    df.rename(columns=col_map, inplace=True)

    # expected standard column order (can be adjusted)
    standard_cols = [
        'gender','age','smoking','yellow_fingers','anxiety','peer_pressure',
        'chronic_disease','fatigue','allergy','wheezing','alcohol','coughing',
        'shortness_of_breath','swallowing_difficulty','chest_pain','lung_cancer'
    ]

    
    # Map gender to 1/0: M/MALE ->1, F/FEMALE->0
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.strip().str.upper().map({
            'M': 1, 'MALE': 1, 'F': 0, 'FEMALE': 0
        }).where(lambda s: ~s.isna(), other=df['gender'])
    

    # For all other expected binary columns, map 2/1 -> 1/0 and YES/NO -> 1/0
    binary_cols = [c for c in standard_cols if c not in ('age','gender')]
    for col in binary_cols:
        if col in df.columns:
            s = df[col]
            # If numeric 1/2 mapping
            if pd.api.types.is_numeric_dtype(s):
                df[col] = s.map({2:1, 1:0}).astype('Int64')
            else:
                # common string forms
                df[col] = s.astype(str).str.strip().str.upper().map({
                    'YES': 1, 'Y': 1, 'TRUE': 1, '1': 0,  # '1' may sometimes represent NO in your encoding; handled below
                    'NO': 0, 'N': 0, 'FALSE': 0, '2': 1
                })
                # if mapping produced NaN, attempt numeric coercion then map 2->1,1->0
                if df[col].isna().any():
                    temp = pd.to_numeric(s, errors='coerce')
                    if temp.notna().any():
                        df[col] = temp.map({2:1,1:0}).astype('Int64')

    # For lung_cancer specifically, allow YES/NO text -> 1/0
    if 'lung_cancer' in df.columns:
        s = df['lung_cancer']
        if not pd.api.types.is_numeric_dtype(s):
            df['lung_cancer'] = s.astype(str).str.strip().str.upper().map({'YES':1,'Y':1,'NO':0,'N':0})
        df['lung_cancer'] = pd.to_numeric(df['lung_cancer'], errors='coerce').map({2:1,1:1,0:0}).astype('Int64')

    # Ensure age is numeric
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Add any missing standard columns as NaN (so merged frame has consistent columns)
    for c in standard_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # return with columns in standard order
    return df[standard_cols]

# Load datasets
d1 = pd.read_csv('Datasets/Lung_Cancer_1.csv')
d2 = pd.read_csv('Datasets/Lung_Cancer_2.csv')
d3 = pd.read_csv('Datasets/Lung_Cancer_3.csv')

# preprocess and merge
merged = merge_datasets(d1, d2, d3)
print("Merged shape:", merged.shape)
print("Column dtypes:")
print(merged.dtypes)

# Save DataFrame into csv
merged.to_csv('Datasets/merged_cancer.csv', index=False)