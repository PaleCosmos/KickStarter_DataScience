import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FILE_INPUT  = "DataSet.csv"
FILE_INPUT_CATEGORY = "Categorical_Attribute.csv"
FILE_OUTPUT = "DataSetOutPut.csv"
HEADER = ['ID','name','category','main_category','currency',
'deadline','goal','launched','pledged','state',
'backers','country','usd pledged','usd_pledged_real','usd_goal_real']


def load_data():
    return pd.read_csv(FILE_INPUT, encoding='utf-8'), pd.read_csv(FILE_INPUT_CATEGORY, encoding='utf-8')

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

#Read Data Set
DataFrame, Categorical_Data = load_data()

#Data Preprocessing
one_hot_encoding = pd.get_dummies(DataFrame[['category','main_category','currency','state','country']],drop_first=True) #one_hot_encdoing (Only categorical attribute)


train_set, test_set= split_train_test(DataFrame,0.25)

DataFrame.plot(kind="scatter",x="usd pledged",y="usd_pledged_real")
plt.show()