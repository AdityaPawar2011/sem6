import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

weather = pd.read_csv("weatherAUS.csv")
# netflix dataset from www.kaggle.com
weather.head(5)

weather.info()

weather.describe(include='all')

print(weather.std())
print('-------------------------------------')
print(weather.median())

weather.shape

weather.isnull().sum()