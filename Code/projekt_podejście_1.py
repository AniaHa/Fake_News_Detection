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

df = pd.read_csv('FactChecking\\train.tsv', delimiter = "\t")

y = (df['label'] == 'pants-fire') * 1
df_X = df.drop('label', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(df_X, y, random_state = 0)

vectorizer = CountVectorizer()
ct = ColumnTransformer([('statment_vect', vectorizer, 'statement')])  # tutaj bedziemy dokladac kolejne transformacje
logistic = LogisticRegression()

p = Pipeline([('column_transformer', ct), ('logistic', logistic)])

p.fit(X_train, y_train)
cross_val_score(p, X_train, y_train, scoring = "roc_auc")

pred = p.predict(X_test)
acc_logistic = np.mean(pred == y_test)
print("test set accuracy", acc_logistic)

# roc curve
scores = p.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, scores)
print(auc(fpr, tpr))

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "k-")
plt.show()

# on train dataset
scores = p.predict_proba(X_train)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, scores)
print(auc(fpr, tpr))

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "k-")
plt.show()

# which words are most important
ind = np.argsort(-abs(p[1].coef_))[:20]
p[0].get_feature_names_out()[ind]