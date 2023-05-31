import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

iris = sns.load_dataset('iris')

iris.head()

iris.describe()

iris.describe(include='object')

iris_groupby = iris.groupby(by='species')

iris_groupby.std()

iris_groupby.mean()

iris_groupby.median()

iris_groupby.min()

iris_groupby.max()

iris_groupby.quantile()