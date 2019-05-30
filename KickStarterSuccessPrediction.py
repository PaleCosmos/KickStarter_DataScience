from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingClassifier
from matplotlib import cm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_curve
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
warnings.simplefilter(action='ignore', category=FutureWarning)
#%matplotlib inline
# params = {'meta-logisticregression__C': [100, 10, 1, 0.1, 0.01]}
np.random.seed(1)

FILE_INPUT = "input/DataSet.csv"

Logistic_Regression_Output = "output/Logistic_Regression.csv"
Naive_Bayes_Output = "output/Naive_Bayes_Output.csv"
Random_Forest_Output = "output/Random_Forest_Output.csv"
K_nearest_neighbors_Output = "output/K_nearest_neighbors.csv"
Bag_of_words_Output = "output/Bag_of_wordsn_Output.csv"
Bag_of_words_with_additional_features_Output = "output/Bag_of_words_with_additional_features.csv"
Comparasion_of_all_result_Output = "output/Comparasion_of_all_result.csv"

switch = [1, 2, 3, 4, 5, 6, 7]
HEADER = ['ID', 'name', 'category', 'main_category', 'currency',
          'deadline', 'goal', 'launched', 'pledged', 'state',
          'backers', 'country', 'usd pledged', 'usd_pledged_real', 'usd_goal_real']


def load_data(INPUT):
    return pd.read_csv(INPUT, encoding='utf-8')


#Load Data
data = load_data(FILE_INPUT)
data.head()


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

def concats(D1,D2,D3,D4,D5,D6,F):return pd.concat([D1,D2,D3,D4,D5,D6,F],axis=1)

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
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, stratify=y, random_state=1)

while(True):
    print('1 -> Logistic Regression',
          '2 -> Naive Bayes',
          '3 -> Random Forest',
          '4 -> K-nearest neighbors',
          '5 -> Bag of words',
          '6 -> Bag of words with additional features',
          '7 -> Comparasion of all result',
          sep='\n')
    #'8 -> Combination of models',

    all_models = {}
    num = int(input("key : "))
    if(not num in switch):
        break
    if(num == 1 or num == 7 or num == 8):
        #Logistic Regression
        model_lr = Pipeline([('preprocess', preprocess_pipeline),
                             ('estimator', LogisticRegression(solver='liblinear', random_state=0))])
        model_lr.fit(X_train, y_train)
        model_lr.score(X_test, y_test)

        y_pred_lr = model_lr.predict(X_test)
        cat1 = concat(pd.DataFrame({'predict_Success': y_pred_lr}).reset_index(drop=True),
                     pd.DataFrame({'Real_Success': y_test}).reset_index(drop=True))
        cat1.to_csv(Logistic_Regression_Output)
        y_pred_proba_lr = model_lr.predict_proba(X_test)[:, 1]

        all_models['lr'] = {}
        all_models['lr']['model'] = model_lr
        all_models['lr']['train_preds'] = model_lr.predict_proba(X_train)[:, 1]
        all_models['lr']['result'] = report_results(
            all_models['lr']['model'], X_test, y_test)
        all_models['lr']['roc_curve'] = get_roc_curve(
            all_models['lr']['model'], X_test, y_test)
        if(num == 1):
            print(cat1)
            MSE(model_lr, X_train, y_train, X_test, y_test, y_pred_lr)
            print(classification_report(y_test, y_pred_lr))
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_lr)
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.plot(fpr, tpr, marker='.')
            plt.show()
    if(num == 2 or num == 7 or num == 8):
        #Naive Bayes
        model_nb = Pipeline([('preprocess', preprocess_pipeline),
                             ('estimator', MultinomialNB())])
        model_nb.fit(X_train, y_train)
        model_nb.score(X_test, y_test)

        y_pred_nb = model_nb.predict(X_test)
        cat2 = concat(pd.DataFrame({'predict_Success': y_pred_nb}).reset_index(drop=True),
                     pd.DataFrame({'Real_Success': y_test}).reset_index(drop=True))
        cat2.to_csv(Naive_Bayes_Output)
        y_pred_proba_nb = model_nb.predict_proba(X_test)[:, 1]

        all_models['nb'] = {}
        all_models['nb']['model'] = model_nb
        all_models['nb']['train_preds'] = model_nb.predict_proba(X_train)[:, 1]
        all_models['nb']['result'] = report_results(
            all_models['nb']['model'], X_test, y_test)
        all_models['nb']['roc_curve'] = get_roc_curve(
            all_models['nb']['model'], X_test, y_test)
        if(num == 2):
            print(cat2)
            MSE(model_nb, X_train, y_train, X_test, y_test, y_pred_nb)

            print(classification_report(y_test, y_pred_nb))
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_nb)
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.plot(fpr, tpr, marker='.')
            plt.show()
    if(num == 3 or num == 7 or num == 8):
        #Random Forest
        model_rf = Pipeline([('preprocess', preprocess_pipeline),
                             ('estimator', RandomForestClassifier(n_estimators=1, random_state=0, n_jobs=-1))])  # estimator 커지면 느려짐
        model_rf.fit(X_train, y_train)
        model_rf.score(X_test, y_test)

        y_pred_rf = model_rf.predict(X_test)
        cat3 = concat(pd.DataFrame({'predict_Success': y_pred_rf}).reset_index(drop=True),
                     pd.DataFrame({'Real_Success': y_test}).reset_index(drop=True))
        cat3.to_csv(Random_Forest_Output)
        y_pred_proba_rf = model_rf.predict_proba(X_test)[:, 1]

        all_models['rf'] = {}
        all_models['rf']['model'] = model_rf
        all_models['rf']['train_preds'] = model_rf.predict_proba(X_train)[:, 1]
        all_models['rf']['result'] = report_results(
            all_models['rf']['model'], X_test, y_test)
        all_models['rf']['roc_curve'] = get_roc_curve(
            all_models['rf']['model'], X_test, y_test)
        if(num == 3):
            print(cat3)
            MSE(model_rf, X_train, y_train, X_test, y_test, y_pred_rf)
            print(classification_report(y_test, y_pred_rf))
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_rf)
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.plot(fpr, tpr, marker='.')
            plt.show()

    if(num == 4 or num == 7 or num == 8):
        #k-nearest neighbors
        model_knn = Pipeline([('preprocess', preprocess_pipeline),
                              ('estimator', KNeighborsClassifier(n_neighbors=3, n_jobs=-1))])

        model_knn.fit(X_train, y_train)
        model_knn.score(X_test, y_test)

        y_pred_knn = model_knn.predict(X_test)
        cat4 = concat(pd.DataFrame({'predict_Success': y_pred_knn}).reset_index(drop=True),
                     pd.DataFrame({'Real_Success': y_test}).reset_index(drop=True))
        cat4.to_csv(K_nearest_neighbors_Output)
        y_pred_proba_knn = model_knn.predict_proba(X_test)[:, 1]

        all_models['knn'] = {}
        all_models['knn']['model'] = model_knn
        all_models['knn']['train_preds'] = model_knn.predict_proba(X_train)[
            :, 1]
        all_models['knn']['result'] = report_results(
            all_models['knn']['model'], X_test, y_test)
        all_models['knn']['roc_curve'] = get_roc_curve(
            all_models['knn']['model'], X_test, y_test)
        if(num == 4):
            print(cat4)
            MSE(model_knn, X_train, y_train, X_test, y_test, y_pred_knn)

            print(classification_report(y_test, y_pred_knn))
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_knn)
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.plot(fpr, tpr, marker='.')
            plt.show()

    if(num == 5 or num == 7 or num == 8):
        #Bag of words
        en_stopwords = set(stopwords.words("english"))
        preprocess_nlp_pipeline = Pipeline(
            [('selector', FunctionTransformer(lambda x: x['name'].fillna(''), validate=False)),
             ('vectorizer', CountVectorizer(stop_words=en_stopwords))
             ]
        )

        preprocess_full_nlp_pipeline = FeatureUnion(
            transformer_list=[('preprocess_nlp_pipeline', preprocess_nlp_pipeline),
                              ('preprocess_base_pipeline', preprocess_base_pipeline)])
        model_nlp = Pipeline([('preprocess', preprocess_nlp_pipeline),
                              ('estimator', LogisticRegression(random_state=0))])
        model_nlp.fit(X_train, y_train)
        model_nlp.score(X_test, y_test)
        y_pred_nlp = model_nlp.predict(X_test)
        cat5 = concat(pd.DataFrame({'predict_Success': y_pred_nlp}).reset_index(drop=True),
                     pd.DataFrame({'Real_Success': y_test}).reset_index(drop=True))
        
        cat5.to_csv(Bag_of_words_Output)
        y_pred_proba_nlp = model_nlp.predict_proba(X_test)[:, 1]
        all_models['nlp'] = {}
        all_models['nlp']['model'] = model_nlp
        all_models['nlp']['train_preds'] = model_nlp.predict_proba(X_train)[
            :, 1]
        all_models['nlp']['result'] = report_results(
            all_models['nlp']['model'], X_test, y_test)
        all_models['nlp']['roc_curve'] = get_roc_curve(
            all_models['nlp']['model'], X_test, y_test)
        if(num == 5):
            print(cat5)
            MSE(model_nlp, X_train, y_train, X_test, y_test, y_pred_nlp)

            print(classification_report(y_test, y_pred_nlp))
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_nlp)
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.plot(fpr, tpr, marker='.')
            plt.show()
    if(num == 6 or num == 7):
        #Bag of words with additional features
        model_mix = Pipeline([('preprocess', preprocess_full_nlp_pipeline),
                              ('estimator', LogisticRegression(random_state=0))])
        model_mix.fit(X_train, y_train)
        model_mix.score(X_test, y_test)
        y_pred_mix = model_mix.predict(X_test)
        cat6 = concat(pd.DataFrame({'predict_Success': y_pred_mix}).reset_index(drop=True),
                     pd.DataFrame({'Real_Success': y_test}).reset_index(drop=True))
        cat6.to_csv(Bag_of_words_with_additional_features_Output)
        y_pred_proba_mix = model_mix.predict_proba(X_test)[:, 1]
        all_models['mix'] = {}
        all_models['mix']['model'] = model_mix
        all_models['mix']['train_preds'] = model_mix.predict_proba(X_train)[
            :, 1]
        all_models['mix']['result'] = report_results(
            all_models['mix']['model'], X_test, y_test)
        all_models['mix']['roc_curve'] = get_roc_curve(
            all_models['mix']['model'], X_test, y_test)
        if(num == 6):
            print(cat6)
            MSE(model_mix, X_train, y_train, X_test, y_test, y_pred_mix)
            print(classification_report(y_test, y_pred_mix))
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_mix)
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.plot(fpr, tpr, marker='.')
            plt.show()
    if(num == 7):
        #Comparasion of all the results
        all_models_name = all_models.keys()
        cat_final = concats(
            pd.DataFrame({'Logistic_Regression':cat1.iloc[:,0]}).reset_index(drop=True),
            pd.DataFrame({'Naive_Bayes':cat2.iloc[:,0]}).reset_index(drop=True),
            pd.DataFrame({'Random_Forest':cat3.iloc[:,0]}).reset_index(drop=True),
            pd.DataFrame({'K_nearest_neighbors':cat4.iloc[:,0]}).reset_index(drop=True),
            pd.DataFrame({'Bag_of_words':cat5.iloc[:,0]}).reset_index(drop=True),
            pd.DataFrame({'Bag_of_words_with_additional_features':cat6.iloc[:,0]}).reset_index(drop=True),
            pd.DataFrame({'Real_Success':y_test}).reset_index(drop=True)
        )
        cat_final.to_csv(Comparasion_of_all_result_Output)
        tmp_list = []
        for mo in all_models_name:
            tmp_list.append(all_models[mo]['result'])
        models_results = pd.DataFrame(
            dict(zip(all_models_name, tmp_list))).transpose()
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
        plt.title('Roc curve')
        plt.legend(loc="lower right")

        plt.show()
        corr_dict = {}
        for mo in tmp_models:
            corr_dict[mo] = all_models[mo]['train_preds']
        kdata_proba = pd.DataFrame(corr_dict)
        corr = kdata_proba.corr()
        corr
        plt.figure(figsize=(14, 8))
        sns.heatmap(corr, cmap="YlGnBu")
        plt.show()
        eclf1 = VotingClassifier(estimators=[('lr', model_lr),
                                             ('rf', model_rf),
                                             ('nb', model_nb),
                                             ('knn', model_knn),
                                             ('nlp', model_nlp)], voting='soft')
        eclf1.fit(X_train, y_train)
        report_results(eclf1, X_test, y_test)
    # if(num == 8):  # 이거안됨
        # lr_stack = LogisticRegression(random_state=0)
        # #lr_stack = RandomForestClassifier()
        # clf_stack = StackingClassifier(
        #     classifiers=[model_nlp, model_lr],
        #     use_probas=True,
        #     average_probas=False,
        #     meta_classifier=lr_stack, verbose=1)
        # clf_stack.fit(X_train, y_train)
        # clf_stack.score(X_test, y_test)
        # report_results(clf_stack, X_test, y_test)
        # grid = GridSearchCV(estimator=clf_stack,
        #                     param_grid=params,
        #                     cv=3,
        #                     refit=True)
        # grid.fit(X_train, y_train)
        # #Optimize the stacked model
        # print('Best parameters: %s' % grid.best_params_)
        # print('Accuracy: %f' % grid.best_score_)
        # report_results(grid, X_test, y_test)
        # eclf1 = VotingClassifier(estimators=[('lr', model_lr),
        #                                      ('rf', model_rf),
        #                                      ('nb', model_nb),
        #                                      ('knn', model_knn),
        #                                      ('nlp', model_nlp)], voting='soft')
        # eclf1.fit(X_train, y_train)
        # report_results(eclf1, X_test, y_test)

print("exited")
