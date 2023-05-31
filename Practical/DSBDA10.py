import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = sns.load_dataset('iris')

dataset.head()

dataset.info()

fig, axes = plt.subplots(2, 2, figsize=(16, 8))

axes[0,0].set_title("Distribution of Sepal Length")
axes[0,0].hist(dataset["sepal_length"]);
axes[0,1].set_title("Distribution of Sepal Width")
axes[0,1].hist(dataset["sepal_width"]);
axes[1,0].set_title("Distribution of Petal Length")
axes[1,0].hist(dataset["petal_length"]);
axes[1,1].set_title("Distribution of Petal Width")
axes[1,1].hist(dataset["petal_width"]);

fig, axes = plt.subplots(2, 2, figsize=(16,9))
axes[0,0].set_title("Distribution of Sepal Length")
sns.boxplot(  y="sepal_length", x= "species", data=dataset,  orient='v' , ax=axes[0, 0])
axes[0,1].set_title("Distribution of Sepal Length")
sns.boxplot(  y="sepal_width", x= "species", data=dataset,  orient='v' , ax=axes[0, 1])
axes[1,0].set_title("Distribution of Sepal Length")
sns.boxplot(  y="petal_length", x= "species", data=dataset,  orient='v' , ax=axes[1, 0])
axes[1,1].set_title("Distribution of Sepal Length")
sns.boxplot(  y="petal_width", x= "species", data=dataset,  orient='v' , ax=axes[1, 1])
plt.show()