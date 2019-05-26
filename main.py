import numpy as np
import pandas as pd
import math


FILE_INPUT  = "DataSet.csv"
FILE_INPUT_CATEGORY = "Categorical_Attribute.csv"
FILE_OUTPUT = "DataSetOutPut.csv"
HEADER = ['ID','name','category','main_category','currency',
'deadline','goal','launched','pledged','state',
'backers','country','usd pledged','usd_pledged_real','usd_goal_real']


def load_data():
    return pd.read_csv(FILE_INPUT, encoding='utf-8'), pd.read_csv(FILE_INPUT_CATEGORY, encoding='utf-8')


def get_mode(df, x,target):    
    #groubby x to target
    group_cat_df = df.groupby([x,target])
    category_df = pd.DataFrame(columns=[x,target,"count"])
        
    get_category = {}
    get_category[np.NaN] = np.NaN
    #make datafraem key and item sorting by count
    for key, item in group_cat_df:
        category_df.loc[len(category_df),:]=[key[0],key[1],item.iloc[:,1].count()]
    category_df = category_df.sort_values(by="count")
    
    #store dictionary sorting by count than store mode value
    for i in range(len(category_df)):
        get_category[category_df.iloc[i,0]] = category_df.iloc[i,1]
    return get_category

def currencyCalculatorToUSD(currency, value):
    switcher ={
        "GBP":1.26,
        "USD":1.0,
        "CAD":0.74,
        "AUD":0.69,
        "NOK":0.11,
        "EUR":1.11,
        "MXN":0.05,
        "SEK":0.1,
        "NZD":0.65,
        "CHF":0.99,
        "DKK":0.15,
        "HKD":0.13,
        "SGD":0.72,
        "JPY":0.0091
    }
    return value*switcher[currency]

df, categorical_Data= load_data()

#print(df)

category_to_main_category = get_mode(df,"category","main_category")
main_category_to_category = get_mode(df,"main_category","category")
currency_to_country = get_mode(df,"currency","country")
country_to_currency = get_mode(df,"country","currency")

#print
group_cat_df = df.groupby(["country"])
category_df = pd.DataFrame(columns=["country","count"])
for key, item in group_cat_df:
    category_df.loc[len(category_df),:]=[key,item.iloc[:,1].count()]
category_df_before = category_df

a = df.info()
isna = pd.isna(df)

#preprocessing
for i in range(len(df)):
    try:
        if isna.loc[i,"category"]:
            df.loc[i,"category"] = main_category_to_category[df.loc[i,"main_category"]]
        
        if isna.loc[i,"main_category"] :
            df.loc[i,"main_category"] = category_to_main_category[df.loc[i,"category"]]
        
        if isna.loc[i,"currency"] or df.loc[i,"currency"] == 'N,0"':
            df.loc[i,"currency"] = country_to_currency[df.loc[i,"country"]];
            
        if isna.loc[i,"country"] or df.loc[i,"country"] == 'N,0"':
            df.loc[i,"country"] = currency_to_country[df.loc[i,"currency"]];
        
        if isna.loc[i,"usd_pledged_real"]:
            df.loc[i,"usd_pledged_real"] = currencyCalculatorToUSD(df.loc[i,"currency"],df.loc[i,"pledged"])
            
        if isna.loc[i,"usd_goal_real"]:
            df.loc[i,"usd_goal_real"] = currencyCalculatorToUSD(df.loc[i,"currency"],df.loc[i,"goal"])
    
    except KeyError:
        pass
    
df['usd_goal_real'].fillna(df.groupby(['main_category', 'category'])['usd_goal_real'].transform("mean"), inplace=True)
df['usd_pledged_real'].fillna(df.groupby(['main_category', 'category'])['usd_pledged_real'].transform("mean"), inplace=True)
#DataFrame.loc[DataFrame['usd_goal_real']<=300, 'usd_goal_real']=np.NaN

#print     
group_cat_df = df.groupby(["country"])
category_df = pd.DataFrame(columns=["country","count"])
for key, item in group_cat_df:
    category_df.loc[len(category_df),:]=[key,item.iloc[:,1].count()]
category_df_after = category_df

df.to_csv("DataSetOutPut.csv");

print(category_df_before)
print(category_df_after)
print(a)
print(df.info())