import seaborn as sns
df = sns.load_dataset('titanic')
df.head()

sns.boxplot(x='sex', y='age', data=df, hue='survived')