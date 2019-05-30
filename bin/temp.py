#%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("sets.csv", encoding='latin1', low_memory=False )

df.head()

print(df.shape[0], 'rows and', df.shape[1], 'columns')

df.columns = [x.strip() for x in df.columns.tolist()]

df[(df['name'].isnull()) | (df['category'].isnull())]
df = df.dropna(axis=0, subset=['name', 'category'])

df = df.iloc[:,:-4]

print(len(df.main_category.unique()), "Main categories")

print(len(df.category.unique()), "sub categories")

sns.set_style('darkgrid')
mains = df.main_category.value_counts().head(15)

x = mains.values
y = mains.index

fig = plt.figure(dpi=100)
ax = fig.add_subplot(111)
ax = sns.barplot(y=y, x=x, orient='h', palette="cool", alpha=0.8)

plt.title('Kickstarter Top 15 Category Count')
plt.show()