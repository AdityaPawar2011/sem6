import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

titanic = sns.load_dataset('titanic')

titanic.info()

titanic.describe()

titanic.shape

sns.histplot(x='fare',data=titanic)
sns.set(rc={'figure.figsize':(5,5)})

sns.displot(x='age',data=titanic,bins=70)
sns.set(rc={'figure.figsize':(5,5)})

sns.catplot(x='survived', data=titanic, kind='count', hue='pclass')
sns.set(rc={'figure.figsize':(5,5)})

sns.factorplot('survived',data=titanic,kind='count',hue='sex')
sns.set(rc={'figure.figsize':(5,5)})