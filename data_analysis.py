"""
Document structure for Diplomski:
- Introductions
- Merging the datasets and preprocessing (Leave M and F unchanged - I belive it's easier to vizualize)
- Exploratory Data Analysis
- Training the models
- Results and Conclusion
"""

import pandas as pd
"""
d1 = pd.read_csv('Datasets/Lung_Cancer_1.csv')
d2 = pd.read_csv('Datasets/Lung_Cancer_2.csv')
d3 = pd.read_csv('Datasets/Lung_Cancer_3.csv')

d1.info()
d2.info()
d3.info()
"""

# Load data and explore
df = pd.read_csv('Datasets/merged_cancer.csv')
df.info()
print(df.head())
print(df.shape)

# Remove nulls and duplicates (already removed when merging)
print(df.isnull().sum())
print(df.duplicated().sum())
# df.drop_duplicates(inplace=True)

# Basic statistics
#print(df.describe(include='all'))
print(df.describe())

"""
WHERE I LEFT OFF:
https://www.kaggle.com/code/casper6290/lung-cancer-prediction-98
https://www.kaggle.com/code/racurry93/lung-cancer-detection-prediction

"""