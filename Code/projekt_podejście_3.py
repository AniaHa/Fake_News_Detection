import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn.utils import estimator_html_repr
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectPercentile, chi2
import spacy
from imblearn.over_sampling import RandomOverSampler


def lemmatize(x):
    l = []
    for d in x:
        l.append(" ".join(t.lemma_ for t in d))
    return l


def lemmatize_pos(x):
    l = []
    for d in x:
        l.append(" ".join(t.lemma_ + " " + t.tag_ for t in d))
    return l


def ner(x):
    l = []
    for d in x:
        l.append(" ".join(e.label_ for e in d.ents))
    return l


nlp = spacy.load("en_core_web_sm")
df = pd.read_csv('FactChecking\\train.tsv', delimiter = "\t")
df = df.fillna("")
df["statement_spacy"] = list(nlp.pipe(df.statement))
df["context_spacy"] = list(nlp.pipe(df.context))
y = (df['label'] == 'pants-fire') * 1
df_X = df.drop('label', axis = 1)
ros = RandomOverSampler(random_state = 42, sampling_strategy = 0.5)
X_train, X_test, y_train, y_test = train_test_split(df_X, y, random_state = 0)
# X_train, y_train = ros.fit_resample(X_train, y_train)

# token_pattern='(?ui)\\b\\w*[a-z]+\\w*\\b' at least one letter
logistic = LogisticRegression(max_iter = 1000)
lasso = LogisticRegression(penalty = 'l1')
vectorizer = TfidfVectorizer()
vectorizer2 = TfidfVectorizer(stop_words = 'english')
Lemmatizer = FunctionTransformer(lemmatize)
LemmatizerPos = FunctionTransformer(lemmatize_pos)
Nerer = FunctionTransformer(ner)
func_trans = FunctionTransformer(accept_sparse = True)
nmf = NMF(n_components = 100)
svd = TruncatedSVD(n_components = 50)
one_hot = OneHotEncoder(drop = 'first', handle_unknown = 'ignore')
union = FeatureUnion([('nmf', nmf), ('id', func_trans)])
p0 = Pipeline([('lemmatizer', LemmatizerPos), ('vect', vectorizer), ('union', union)])
p1 = Pipeline([('vect', vectorizer2), ('union', union)])
p2 = Pipeline([('lemmatizer', LemmatizerPos), ('vect', vectorizer2), ('union', union)])
p3 = Pipeline([('ner', Nerer), ('vect', vectorizer), ('union', union)])
ct = ColumnTransformer([('statement_lemma', p0, 'statement_spacy'), ('encoder_party', one_hot, ['party']),
                        ('subject_vect', p1, 'subject'), ('encoder_state', one_hot, ['state']),
                        ('encoder_speaker', one_hot, ['speaker']), ('context_vect', p1, 'context'),
                        ('context_lemma', p2, 'context_spacy'), ('encoder_speaker_job', one_hot, ['speaker_job']),
                        ('statement_ner', p3, 'statement_spacy')])
selector = SelectPercentile(percentile = 90)
p = Pipeline([('column_transformer', ct), ('feature_selection', selector),
              ('logistic', logistic)])
p_lasso = Pipeline([('column_transformer', ct), ('feature_selection', selector),
                    ('logistic', lasso)])

p_lasso.fit(X_train, y_train)
p_lasso[2].coef_

p.fit(X_train, y_train)

cv = StratifiedKFold(5) # trzyma stałe proporcje klas
np.mean(cross_val_score(p, df_X, y, scoring = "roc_auc", cv = cv))

pred_nmf = p.predict(X_test)
acc_logistic_nmf = np.mean(pred_nmf == y_test)
print("test set accuracy", acc_logistic_nmf)

# ROC curve
scores = p.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, scores)
auc_score = auc(fpr, tpr)
print(auc_score)

with open('model.html', 'w', encoding = 'utf-8') as f:
    f.write(estimator_html_repr(p))
