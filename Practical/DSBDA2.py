import pandas as pd
import numpy as np
df = pd.read_csv("StudentsPerformance.csv")
df.head()

df.isnull().sum()

series = df['math score'].isnull()
df[series]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

m_avg = df['math score'].mean()
df['math score'].fillna(value = m_avg, inplace=True)

df['math score']


# to delete null values we use dropna
df.dropna()

import matplotlib.pyplot as plt
df.boxplot(['math score', 'reading score', 'writing score'])