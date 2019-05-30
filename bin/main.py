import numpy as np
import pandas as pd
import math
from scipy import stats
from matplotlib import pyplot as plt
from mlxtend.preprocessing import minmax_scaling
import seaborn as sns


FILE_INPUT  = "DataSetOutput.csv"
HEADER = ['ID','name','category','main_category','currency',
'deadline','goal','launched','pledged','state',
'backers','country','usd pledged','usd_pledged_real','usd_goal_real']

def load_data():
    return pd.read_csv(FILE_INPUT, encoding='utf-8')

df = load_data()

#Scaling
usd_goal = df.usd_goal_real
scaled_data = minmax_scaling(usd_goal,columns=[0])

fig, ax=plt.subplots(1,2)
sns.distplot(df.usd_goal_real, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")


#Normalization
index_of_positive_pledges = df.usd_pledged_real > 0


positive_pledges = df.usd_pledged_real.loc[index_of_positive_pledges]


normalized_pledges = stats.boxcox(positive_pledges)[0]

fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalized data")

plt.show()