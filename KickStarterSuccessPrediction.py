from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingClassifier
from scipy import stats
from mlxtend.preprocessing import minmax_scaling
from sklearn.model_selection import cross_val_score
from matplotlib import cm
from sklearn import model_selection
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, auc
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, LabelBinarizer
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import math
import os
os.environ["PATH"] += os.pathsep + \
    'C:\\program Files (x86)\\Graphviz2.38\\bin\\'

warnings.simplefilter(action='ignore', category=FutureWarning)
#%matplotlib inline
# params = {'meta-logisticregression__C': [100, 10, 1, 0.1, 0.01]}
np.random.seed(1)
seed = 7

FILE_INPUT = "input/DataSet.csv"

Logistic_Regression_Output = "output/Logistic_Regression.csv"
Decision_Tree_Output = "output/Decision_Tree_Output.csv"
Random_Forest_Output = "output/Random_Forest_Output.csv"
K_nearest_neighbors_Output = "output/K_nearest_neighbors.csv"

Logistic_Regression_Output_bagging = "output/Logistic_Regression.csv"
Decision_Tree_Output_bagging = "output/Decision_Tree_Output.csv"
Random_Forest_Output_bagging = "output/Random_Forest_Output.csv"
K_nearest_neighbors_Output_bagging = "output/K_nearest_neighbors.csv"

Comparasion_of_all_result_Output = "output/Comparasion_of_all_result.csv"

switch = [1, 2, 3, 4, 5]
HEADER = ['ID', 'name', 'category', 'main_category', 'currency',
          'deadline', 'goal', 'launched', 'pledged', 'state',
          'backers', 'country', 'usd pledged', 'usd_pledged_real', 'usd_goal_real']


def load_data(INPUT):
    return pd.read_csv(INPUT, encoding='utf-8')


#Load Data
data = load_data(FILE_INPUT)
data = data.loc[:10000, :]
data.head()

#Define Class
class MultiColumnLabelEncoder(TransformerMixin):
    def __init__(self):
        self.d = defaultdict(LabelEncoder)

    def transform(self, X, **transform_params):

        X = X.fillna('NaN')
        transformed = X.apply(self._transform_func)
        return transformed

    def fit(self, X, y=None, **fit_params):
        X = X.fillna('NaN')
        X.apply(self._fit_func)
        return self

    def _transform_func(self, x):
        return self.d[x.name].transform(x)

    def _fit_func(self, x):
        return self.d[x.name].fit(x)


def get_categorical_data(x):
    return x[['category', 'main_category', 'currency', 'country']]


def get_name_lenght_feature(x):
    return x['name'].str.len().fillna(0).to_frame()


def get_duration_feature(x):
    return (pd.to_datetime(x['deadline']) - pd.to_datetime(x['launched'])).dt.days.to_frame()


def get_deadline_month_feature(x):
    return pd.to_datetime(x['deadline']).dt.month.to_frame()


def get_deadline_weekday_feature(x):
    return pd.to_datetime(x['deadline']).dt.weekday.to_frame()


def get_launched_month_feature(x):
    return pd.to_datetime(x['launched']).dt.month.to_frame()


def get_launched_weekday_feature(x):
    return pd.to_datetime(x['launched']).dt.weekday.to_frame()


def concat(D1, D2): return pd.concat([D1, D2], axis=1)


def concats(D1, D2, D3, D4, D5, D6, F): return pd.concat(
    [D1, D2, D3, D4, D5, D6, F], axis=1)


def MSE(fit, train_set_x, train_set_y, test_set_x, test_set_y, p):
    #Mean Squared Error
    MSE = np.sqrt(np.mean((p - test_set_y) ** 2))
    RMSE = math.sqrt(MSE)
    print("MSE: %.2f" % MSE)
    print("RMSE: %.2f" % RMSE)
    # Explained variance score: 1 is perfect prediction
    print('Train Variance score: %.2f' % fit.score(train_set_x, train_set_y))
    print('Test Variance score: %.2f' % fit.score(test_set_x, test_set_y))


preprocess_base_pipeline = FeatureUnion(
    transformer_list=[
        ('name_length', Pipeline([
            ('selector', FunctionTransformer(
                get_name_lenght_feature, validate=False))
        ])),
        ('duration_feature', Pipeline([
            ('selector', FunctionTransformer(
                get_duration_feature, validate=False))
        ])),
        ('deadline_month', Pipeline([
            ('selector', FunctionTransformer(
                get_deadline_month_feature, validate=False))
        ])),
        ('deadline_weekday', Pipeline([
            ('selector', FunctionTransformer(
                get_deadline_weekday_feature, validate=False))
        ])),
        ('launched_month', Pipeline([
            ('selector', FunctionTransformer(
                get_launched_month_feature, validate=False))
        ])),
        ('launched_weekday', Pipeline([
            ('selector', FunctionTransformer(
                get_launched_weekday_feature, validate=False))
        ]))
    ])
preprocess_pipeline = FeatureUnion(
    transformer_list=[
        ('cat_features', Pipeline([
            ('selector', FunctionTransformer(
                get_categorical_data, validate=False)),
            ('encoder', MultiColumnLabelEncoder())
        ])),

        ('name_length', Pipeline([
            ('preprocess_base_pipeline', preprocess_base_pipeline)
        ])),

    ])


def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {'auc': auc, 'f1': f1, 'acc': acc,
              'precision': prec, 'recall': rec}
    return result


def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)

    return fpr, tpr


X = preprocess_pipeline.fit_transform(data)
y = (data['state'] == 'successful').astype('int')

sns.heatmap(pd.DataFrame(X).corr(), cmap='Blues')
plt.title('Feature Correlations')
plt.savefig('image/feature.png')
plt.close()
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, stratify=y, random_state=1)

#loop
while(True):
    print('1 -> Logistic Regression',
          '2 -> Decision Tree',
          '3 -> Random Forest',
          '4 -> K-nearest neighbors',
          '5 -> Comparasion of all result',
          sep='\n')

    all_models = {}
    normal_models = {}
    bagging_models = {}

    #get flag
    num = int(input("key : "))

    if(not num in switch):
        break

     #Logistic Regression
    if(num == 1 or num == 5):
       
        model_lr = Pipeline([('preprocess', preprocess_pipeline),
                             ('estimator', LogisticRegression(solver='liblinear', random_state=0))])

        model_lr.fit(X_train, y_train)
        model_lr.score(X_test, y_test)

        y_pred_lr = model_lr.predict(X_test)
        cat_lr = concat(pd.DataFrame({'predict_Success': y_pred_lr}).reset_index(drop=True),
                        pd.DataFrame({'Real_Success': y_test}).reset_index(drop=True))
        cat_lr.to_csv(Logistic_Regression_Output)
        y_pred_proba_lr = model_lr.predict_proba(X_test)[:, 1]

        all_models['lr'] = {}
        all_models['lr']['model'] = model_lr

        all_models['lr']['train_preds'] = model_lr.predict_proba(X_train)[:, 1]

        all_models['lr']['result'] = report_results(
            all_models['lr']['model'], X_test, y_test)

        all_models['lr']['roc_curve'] = get_roc_curve(
            all_models['lr']['model'], X_test, y_test)

        #print(cat_lr)
        MSE(model_lr, X_train, y_train, X_test, y_test, y_pred_lr)
        print(classification_report(y_test, y_pred_lr))
        fpr_1, tpr_1, thresholds = roc_curve(y_test, y_pred_proba_lr)

        # -----------------------------------------------------------------------------
        num_trees = 10

        model_lrb = Pipeline([('preprocess', preprocess_pipeline),
                              ('estimator', BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=num_trees, random_state=seed))])
        model_lrb.fit(X_train, y_train)
        model_lrb.score(X_test, y_test)
        y_preds_lrb = model_lrb.predict(X_test)
        cat_lrb = concat(pd.DataFrame({'predict_Success': y_preds_lrb}).reset_index(drop=True),
                         pd.DataFrame({'Real_Success': y_test}).reset_index(drop=True))
        cat_lrb.to_csv(Logistic_Regression_Output_bagging)
        y_preds_proba_lrb = model_lrb.predict_proba(X_test)[:, 1]

        all_models["lrb"] = {}
        all_models["lrb"]['model'] = model_lrb
        all_models["lrb"]['train_preds'] = model_lrb.predict_proba(X_train)[
            :, 1]
        all_models["lrb"]['result'] = report_results(
            all_models["lrb"]['model'], X_test, y_test)
        all_models["lrb"]['roc_curve'] = get_roc_curve(
            all_models["lrb"]['model'], X_test, y_test)
        MSE(model_lrb, X_train, y_train, X_test, y_test, y_preds_lrb)

        normal_models['lr'] = {}
        bagging_models['lrb'] = {}
        normal_models['lr'] = all_models['lr']
        bagging_models['lrb'] = all_models['lrb']
        #print(cat_lrb)
        print(classification_report(y_test, y_preds_lrb))
        fpr_1c, tpr_1c, thresholds = roc_curve(y_test, y_preds_proba_lrb)

        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(fpr_1, tpr_1, marker='.', color='blue')
        plt.plot(fpr_1c, tpr_1c, marker='.', color='red')
        plt.savefig('image/Logistic_Regression.png')
        if(num == 1):
            plt.show()
        plt.close()
    #Decesion Tree   
    if(num == 2 or num == 5):
        
        model_dt = Pipeline([('preprocess', preprocess_pipeline),
                             ('estimator', DecisionTreeClassifier())])
        model_dt.fit(X_train, y_train)
        model_dt.score(X_test, y_test)

        y_pred_dt = model_dt.predict(X_test)
        cat_dt = concat(pd.DataFrame({'predict_Success': y_pred_dt}).reset_index(drop=True),
                        pd.DataFrame({'Real_Success': y_test}).reset_index(drop=True))
        cat_dt.to_csv(Decision_Tree_Output)
        y_pred_proba_dt = model_dt.predict_proba(X_test)[:, 1]

        all_models['dt'] = {}
        all_models['dt']['model'] = model_dt
        all_models['dt']['train_preds'] = model_dt.predict_proba(X_train)[:, 1]
        all_models['dt']['result'] = report_results(
            all_models['dt']['model'], X_test, y_test)
        all_models['dt']['roc_curve'] = get_roc_curve(
            all_models['dt']['model'], X_test, y_test)

        #print(cat_dt)
        MSE(model_dt, X_train, y_train, X_test, y_test, y_pred_dt)

        print(classification_report(y_test, y_pred_dt))
        fpr_2, tpr_2, thresholds = roc_curve(y_test, y_pred_proba_dt)

        # -----------------------------------------------------------------------------
        num_trees = 10

        model_dtb = Pipeline([('preprocess', preprocess_pipeline),
                              ('estimator', BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=num_trees, random_state=seed))])
        model_dtb.fit(X_train, y_train)
        model_dtb.score(X_test, y_test)
        y_preds_dtb = model_dtb.predict(X_test)
        cat_dtb = concat(pd.DataFrame({'predict_Success': y_preds_dtb}).reset_index(drop=True),
                         pd.DataFrame({'Real_Success': y_test}).reset_index(drop=True))
        cat_dtb.to_csv(Decision_Tree_Output_bagging)
        y_preds_proba_dtb = model_dtb.predict_proba(X_test)[:, 1]

        all_models["dtb"] = {}
        all_models["dtb"]['model'] = model_dtb
        all_models["dtb"]['train_preds'] = model_dtb.predict_proba(X_train)[
            :, 1]
        all_models["dtb"]['result'] = report_results(
            all_models["dtb"]['model'], X_test, y_test)
        all_models["dtb"]['roc_curve'] = get_roc_curve(
            all_models["dtb"]['model'], X_test, y_test)

        #print(cat_dtb)
        MSE(model_dtb, X_train, y_train, X_test, y_test, y_preds_dtb)
        normal_models['dt'] = {}
        bagging_models['dtb'] = {}
        normal_models['dt'] = all_models['dt']
        bagging_models['dtb'] = all_models['dtb']
        print(classification_report(y_test, y_preds_dtb))
        fpr_2c, tpr_2c, thresholds = roc_curve(y_test, y_preds_proba_dtb)

        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(fpr_2, tpr_2, marker='.', color='blue')
        plt.plot(fpr_2c, tpr_2c, marker='.', color='red')
        plt.savefig('image/Decision_Tree.png')
        if(num == 2):
            plt.show()
        plt.close()
    
    #Random Forest
    if(num == 3 or num == 5):
        
        model_rf = Pipeline([('preprocess', preprocess_pipeline),
                             ('estimator', RandomForestClassifier(n_estimators=1, random_state=0, n_jobs=-1))])  # estimator 커지면 느려짐
        model_rf.fit(X_train, y_train)
        model_rf.score(X_test, y_test)

        y_pred_rf = model_rf.predict(X_test)
        cat_rf = concat(pd.DataFrame({'predict_Success': y_pred_rf}).reset_index(drop=True),
                        pd.DataFrame({'Real_Success': y_test}).reset_index(drop=True))
        cat_rf.to_csv(Random_Forest_Output)
        y_pred_proba_rf = model_rf.predict_proba(X_test)[:, 1]

        all_models['rf'] = {}
        all_models['rf']['model'] = model_rf
        all_models['rf']['train_preds'] = model_rf.predict_proba(X_train)[:, 1]
        all_models['rf']['result'] = report_results(
            all_models['rf']['model'], X_test, y_test)
        all_models['rf']['roc_curve'] = get_roc_curve(
            all_models['rf']['model'], X_test, y_test)

        #print(cat_rf)
        MSE(model_rf, X_train, y_train, X_test, y_test, y_pred_rf)
        print(classification_report(y_test, y_pred_rf))
        fpr_3, tpr_3, thresholds = roc_curve(y_test, y_pred_proba_rf)

        # -----------------------------------------------------------------------------
        num_trees = 10

        model_rfb = Pipeline([('preprocess', preprocess_pipeline),
                              ('estimator', BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=num_trees, random_state=seed))])
        model_rfb.fit(X_train, y_train)
        model_rfb.score(X_test, y_test)
        y_preds_rfb = model_rfb.predict(X_test)
        cat_rfb = concat(pd.DataFrame({'predict_Success': y_preds_rfb}).reset_index(drop=True),
                         pd.DataFrame({'Real_Success': y_test}).reset_index(drop=True))
        cat_rfb.to_csv(Random_Forest_Output)
        y_preds_proba_rfb = model_rfb.predict_proba(X_test)[:, 1]

        all_models["rfb"] = {}
        all_models["rfb"]['model'] = model_rfb
        all_models["rfb"]['train_preds'] = model_rfb.predict_proba(X_train)[
            :, 1]
        all_models["rfb"]['result'] = report_results(
            all_models["rfb"]['model'], X_test, y_test)
        all_models["rfb"]['roc_curve'] = get_roc_curve(
            all_models["rfb"]['model'], X_test, y_test)
        #print(cat_rfb)
        MSE(model_rfb, X_train, y_train, X_test, y_test, y_preds_rfb)

        normal_models['rf'] = {}
        bagging_models['rfb'] = {}
        normal_models['rf'] = all_models['rf']
        bagging_models['rfb'] = all_models['rfb']
        print(classification_report(y_test, y_preds_rfb))
        fpr_3c, tpr_3c, thresholds = roc_curve(y_test, y_preds_proba_rfb)

        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(fpr_3, tpr_3, marker='.', color='blue')
        plt.plot(fpr_3c, tpr_3c, marker='.', color='red')
        plt.savefig("image/Random_Forest.png")
        if(num == 3):
            plt.show()
        plt.close()
    #k-nearest neighbors
    if(num == 4 or num == 5):
        
        model_knn = Pipeline([('preprocess', preprocess_pipeline),
                              ('estimator', KNeighborsClassifier(n_neighbors=3, n_jobs=-1))])

        model_knn.fit(X_train, y_train)
        model_knn.score(X_test, y_test)

        y_pred_knn = model_knn.predict(X_test)
        cat_knn = concat(pd.DataFrame({'predict_Success': y_pred_knn}).reset_index(drop=True),
                         pd.DataFrame({'Real_Success': y_test}).reset_index(drop=True))
        cat_knn.to_csv(K_nearest_neighbors_Output)
        y_pred_proba_knn = model_knn.predict_proba(X_test)[:, 1]

        all_models['knn'] = {}
        all_models['knn']['model'] = model_knn
        all_models['knn']['train_preds'] = model_knn.predict_proba(X_train)[
            :, 1]
        all_models['knn']['result'] = report_results(
            all_models['knn']['model'], X_test, y_test)
        all_models['knn']['roc_curve'] = get_roc_curve(
            all_models['knn']['model'], X_test, y_test)

        #print(cat_knn)
        MSE(model_knn, X_train, y_train, X_test, y_test, y_pred_knn)

        print(classification_report(y_test, y_pred_knn))
        fpr_4, tpr_4, thresholds = roc_curve(y_test, y_pred_proba_knn)

        # -----------------------------------------------------------------------------
        num_trees = 10

        model_knnb = Pipeline([('preprocess', preprocess_pipeline),
                               ('estimator', BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=num_trees, random_state=seed))])
        model_knnb.fit(X_train, y_train)
        model_knnb.score(X_test, y_test)
        y_preds_knnb = model_knnb.predict(X_test)
        cat_knnb = concat(pd.DataFrame({'predict_Success': y_preds_knnb}).reset_index(drop=True),
                          pd.DataFrame({'Real_Success': y_test}).reset_index(drop=True))
        cat_knnb.to_csv(K_nearest_neighbors_Output)
        y_preds_proba_knnb = model_knnb.predict_proba(X_test)[:, 1]

        all_models["knnb"] = {}
        all_models["knnb"]['model'] = model_knnb
        all_models["knnb"]['train_preds'] = model_knnb.predict_proba(X_train)[
            :, 1]
        all_models["knnb"]['result'] = report_results(
            all_models["knnb"]['model'], X_test, y_test)
        all_models["knnb"]['roc_curve'] = get_roc_curve(
            all_models["knnb"]['model'], X_test, y_test)

        #print(cat_knnb)
        MSE(model_knnb, X_train, y_train, X_test, y_test, y_preds_knnb)
        normal_models['knn'] = {}
        bagging_models['knnb'] = {}
        normal_models['knn'] = all_models['knn']
        bagging_models['knnb'] = all_models['knnb']
        print(classification_report(y_test, y_preds_knnb))
        fpr_4c, tpr_4c, thresholds = roc_curve(y_test, y_preds_proba_knnb)

        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(fpr_4, tpr_4, marker='.', color='blue')
        plt.plot(fpr_4c, tpr_4c, marker='.', color='red')
        plt.savefig('image/K_nearest.png')
        if(num == 4):
            plt.show()
        plt.close()
        
    #comparise all output
    if(num == 5):
        eclf1 = VotingClassifier(estimators=[('lr', model_lr),
                                             ('rf', model_rf),
                                             ('nb', model_dt),
                                             ('knn', model_knn)], voting='soft')

        eclf1.fit(X_train, y_train)

        report_results(eclf1, X_test, y_test)

        all_models['bagging'] = {}
        all_models['bagging']['model'] = eclf1
        all_models['bagging']['train_preds'] = eclf1.predict_proba(X_train)[
            :, 1]
        all_models['bagging']['result'] = report_results(
            all_models['bagging']['model'], X_test, y_test)
        all_models['bagging']['roc_curve'] = get_roc_curve(
            all_models['bagging']['model'], X_test, y_test)

        normal_models['bagging'] = {}
        normal_models['bagging'] = all_models['bagging']

        #Comparasion of all the results
        all_models_name = all_models.keys()
        normal_models_name = normal_models.keys()
        bagging_models_name = bagging_models.keys()

    # -----------------------------------------------------------------------------
        tmp_list = []
        for mo in normal_models_name:
            tmp_list.append(normal_models[mo]['result'])
        models_results = pd.DataFrame(
            dict(zip(normal_models_name, tmp_list))).transpose()
        models_results = models_results.sort_values(['auc'], ascending=False)
        models_results
        tmp_models = models_results.index

        colors = cm.rainbow(np.linspace(0.0, 1.0, len(tmp_models)))

        plt.figure(figsize=(14, 8))
        lw = 2

        for mo, color in zip(tmp_models, colors):
            fpr, tpr = all_models[mo]['roc_curve']
            plt.plot(fpr, tpr, color=color,
                     lw=lw, label='{} (auc = {:.4f})'.format(mo, all_models[mo]['result']['auc']))

        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Roc curve_normal')
        plt.legend(loc="lower right")
        plt.savefig('image/normal.png')
        plt.close()
        #plt.show()
     # -----------------------------------------------------------------------------
        tmp_list = []
        for mo in bagging_models_name:
            tmp_list.append(bagging_models[mo]['result'])
        models_results = pd.DataFrame(
            dict(zip(bagging_models_name, tmp_list))).transpose()
        models_results = models_results.sort_values(['auc'], ascending=False)
        models_results
        tmp_models = models_results.index

        colors = cm.rainbow(np.linspace(0.0, 1.0, len(tmp_models)))

        plt.figure(figsize=(14, 8))
        lw = 2

        for mo, color in zip(tmp_models, colors):
            fpr, tpr = all_models[mo]['roc_curve']
            plt.plot(fpr, tpr, color=color,
                     lw=lw, label='{} (auc = {:.4f})'.format(mo, all_models[mo]['result']['auc']))

        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Roc curve_bagging')
        plt.legend(loc="lower right")
        plt.savefig('image/bagging.png')
        plt.close()
        #plt.show()
    # -----------------------------------------------------------------------------

        corr_dict = {}
        for mo in tmp_models:
            corr_dict[mo] = all_models[mo]['train_preds']
        kdata_proba = pd.DataFrame(corr_dict)
        corr = kdata_proba.corr()
        corr
        plt.figure(figsize=(14, 8))
        sns.heatmap(corr, cmap="YlGnBu")
        plt.savefig('image/last.png')
        #plt.show()
        plt.close()


print("exited")
