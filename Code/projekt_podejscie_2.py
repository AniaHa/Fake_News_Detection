import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn.utils import estimator_html_repr

df = pd.read_csv('FactChecking\\train.tsv', delimiter = "\t")

y = (df['label'] == 'pants-fire') * 1
df_X = df.drop('label', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(df_X, y, random_state = 0)

# without transformation
vectorizer = TfidfVectorizer(stop_words = 'english')
ct = ColumnTransformer([('statment_vect', vectorizer, 'statement')])  # tutaj bedziemy dokladac kolejne transformacje
logistic = LogisticRegression()

p = Pipeline([('column_transformer', ct), ('logistic', logistic)])

p.fit(X_train, y_train)
cross_val_score(p, X_train, y_train, scoring = "roc_auc")

pred = p.predict(X_test)
acc_logistic = np.mean(pred == y_test)
print("test set accuracy", acc_logistic)

scores = p.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, scores)
auc_n = auc(fpr, tpr)

with open('my_estimator.html', 'w', encoding='utf-8') as f:
    f.write(estimator_html_repr(p))

# NMF
n_c = 20
nmf = NMF(n_components = n_c, max_iter = 100)
p0_nmf = Pipeline([('vect', vectorizer), ('nmf', nmf)])
ct_nmf = ColumnTransformer([('statment_vect', p0_nmf, 'statement')])
p_nmf = Pipeline([('column_transformer', ct_nmf), ('logistic', logistic)])

p_nmf.fit(X_train, y_train)
cross_val_score(p_nmf, X_train, y_train, scoring = "roc_auc")

pred_nmf = p_nmf.predict(X_test)
acc_logistic_nmf = np.mean(pred_nmf == y_test)
print("test set accuracy", acc_logistic_nmf)

# ROC curve
scores = p_nmf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, scores)
auc_nmf = auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "k-")
plt.show()

# top 10 words in topic
# pola z podkreśleniem na końcu powstają w trakcie fitowania modelu
k = 10
words_nmf = p_nmf[0].named_transformers_['statment_vect'][0].get_feature_names_out()
top10_nmf = p_nmf[0].named_transformers_['statment_vect'][1].components_[k].argsort()[-10:]
print(words_nmf[top10_nmf])

with open('my_estimator_nmf.html', 'w', encoding='utf-8') as f:
    f.write(estimator_html_repr(p_nmf))

# TruncatedSVD
svd = TruncatedSVD(n_components = n_c)
p0_svd = Pipeline([('vect', vectorizer), ('nmf', svd)])
ct_svd = ColumnTransformer([('statment_vect', p0_svd, 'statement')])
p_svd = Pipeline([('column_transformer', ct_svd), ('logistic', logistic)])

p_svd.fit(X_train, y_train)
cross_val_score(p_svd, X_train, y_train, scoring = "roc_auc")

pred_svd = p_svd.predict(X_test)
acc_logistic_svd = np.mean(pred_svd == y_test)
print("test set accuracy", acc_logistic_svd)

# ROC curve
scores = p_svd.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, scores)
auc_svd = auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "k-")
plt.show()

# top 10 words in topic
k = 10
words_svd = p_svd[0].named_transformers_['statment_vect'][0].get_feature_names_out()
top10_svd = p_svd[0].named_transformers_['statment_vect'][1].components_[k].argsort()[-10:]
print(words_svd[top10_svd])

# LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components = n_c)
p0_lda = Pipeline([('vect', vectorizer), ('nmf', lda)])
ct_lda = ColumnTransformer([('statment_vect', p0_lda, 'statement')])
p_lda = Pipeline([('column_transformer', ct_lda), ('logistic', logistic)])

p_lda.fit(X_train, y_train)
cross_val_score(p_lda, X_train, y_train, scoring = "roc_auc")

pred_lda = p_lda.predict(X_test)
acc_logistic_lda = np.mean(pred_lda == y_test)
print("test set accuracy", acc_logistic_lda)

# ROC curve
scores = p_lda.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, scores)
auc_lda = auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "k-")
plt.show()

# top 10 words in topic
k = 10
words_lda = p_lda[0].named_transformers_['statment_vect'][0].get_feature_names_out()
top10_lda = p_lda[0].named_transformers_['statment_vect'][1].components_[k].argsort()[-10:]
print(words_lda[top10_lda])

# Comparison
d = {'Transformation': ['NMF', 'SVD', 'LDA', 'no transformation'], 'Accuracy': [acc_logistic_nmf, acc_logistic_svd,
                                                                                acc_logistic_lda, acc_logistic],
     'AUC': [auc_nmf, auc_svd, auc_lda, auc_n], 'Top 10 words': [words_nmf[top10_nmf], words_svd[top10_svd],
                                                                 words_lda[top10_lda], 'None']}
comparison = pd.DataFrame(data = d)

# one hot encoder: handle_error - kategorie które są w testowym ale nie było w treningowym
# one hot encoder bywa problematyczny dla dużej liczby poziomów
# rozwiązanie: FeatureHasher (tablice hashujące) - część kategorii dostanie ten sam współczynnik dla za malej liczby
# indykatorów
# feature union - kilka transformacji na jednej zmiennej i zlączenie ich wyników, aby zrobic identyczościową
# tranformacje nalezy użyc function transformera

