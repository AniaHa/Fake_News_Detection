import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn.utils import estimator_html_repr
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from sklearn.feature_selection import SelectPercentile, chi2, mutual_info_regression, RFE
import spacy
import gensim
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.base import BaseEstimator, TransformerMixin

wyniki = []


class Doc2VecVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, vector_size=100, epochs=10, alpha=0.025,
                 min_alpha=0.025):
        self.vector_size = vector_size
        self.epochs = epochs
        self.alpha = alpha
        self.min_alpha = min_alpha

    def fit(self, X, y=None):
        if hasattr(X, "iloc"):
            X0 = X.iloc[0]
        else:
            X0 = X[0]
        if not isinstance(X0, TaggedDocument):
            X = [TaggedDocument(list(gensim.utils.tokenize(d)), [str(i)]) for i, d in enumerate(X)]
        model = Doc2Vec(vector_size = self.vector_size,
                        alpha = self.alpha, min_alpha = self.min_alpha)
        model.build_vocab(X)
        model.train(X, epochs = self.epochs, total_examples = model.corpus_count)
        self.model_ = model
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.model_.dv.vectors

    def transform(self, X, copy=True):
        assert self.model_ is not None, 'model is not fitted'
        return np.array([self.model_.infer_vector(list(gensim.utils.tokenize(d))) for d in X])


# funkcje pomocnicze
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


def pos(x):
    l = []
    for d in x:
        l.append(" ".join(t.tag_ for t in d))
    return l


def ner(x):
    l = []
    for d in x:
        l.append(" ".join(e.label_ for e in d.ents))
    return l


def is_digit(x):
    l = []
    emp_str = []
    for m in x:
        if m.isdigit():
            emp_str.append(m)
    if not emp_str:
        l.append(0)
    else:
        l.append(1)
    return l


def extract_text_features(s):
    s = s.astype('str')
    n = s.str.len().values
    n_w = s.str.split().str.len()
    avg_w_len = n.astype(float) / n_w
    # num = s.str.extract(r'(\d+)').fillna(0)
    # num = (num != 0) * 1
    # sym = s.str.extract(r'([/#]\S*)+').fillna(0)
    # sym = (sym != 0) * 1
    return np.column_stack([n, n_w, avg_w_len])


def text_len(s):
    s = s.astype('str')
    n_w = s.str.split().str.len()
    return n_w.values.reshape(-1, 1)


def avg_word_len(s):
    s = s.astype('str')
    n = s.str.len().values
    n_w = s.str.split().str.len()
    avg_w_len = n.astype(float) / n_w
    return avg_w_len.values.reshape(-1, 1)


def word_len(s):
    s = s.astype('str')
    n = s.str.len().values
    return n.reshape(-1, 1)


# wczytanie zbioru danych
nlp = spacy.load("en_core_web_sm")
df = pd.read_csv('FactChecking\\train.tsv', delimiter = "\t")
df = df.fillna(" ")
df["statement_spacy"] = list(nlp.pipe(df.statement))
df["context_spacy"] = list(nlp.pipe(df.context))
df["speaker_job_spacy"] = list(nlp.pipe(df.speaker_job))
df["speaker_spacy"] = list(nlp.pipe(df.speaker))
df["subject_spacy"] = list(nlp.pipe(df.subject))
df['speaker_job_none'] = (df['speaker_job'] == ' ')*1
y = (df['label'] == 'pants-fire') * 1
df_X = df.drop('label', axis = 1)
# X_train, X_test, y_train, y_test = train_test_split(df_X, y, random_state = 0)
final_test = pd.read_csv('FactChecking\\test_noy.tsv', delimiter = "\t")

# Pipeline
logistic = LogisticRegression(max_iter = 1000, C = 0.9)
vectorizer = TfidfVectorizer(token_pattern = '(?u)\\b\\w+\\b', min_df=0.0001, max_df=0.9)
# '(?u)\\b\\w+\\b' lub '(?ui)\\b\\w*[a-z]+\\w*\\b'
vectorizer2 = TfidfVectorizer(stop_words = 'english', min_df = 0.001)
vectorizer3 = CountVectorizer()
Lemmatizer = FunctionTransformer(lemmatize)
LemmatizerPos = FunctionTransformer(lemmatize_pos)
Nerer = FunctionTransformer(ner)
Pos = FunctionTransformer(pos)
TextFeatures = FunctionTransformer(extract_text_features)
TextLen = FunctionTransformer(text_len)
func_trans = FunctionTransformer(accept_sparse = True)
nmf = NMF()
svd = TruncatedSVD(n_components = 50)
one_hot = OneHotEncoder(drop = 'first', handle_unknown = 'ignore')
union = FeatureUnion([('svd', svd), ('id', func_trans)])
union2 = FeatureUnion([('nmf', nmf), ('id', func_trans)])
p0 = Pipeline([('lemmatizer', LemmatizerPos), ('vect', vectorizer), ('union', union)])
p1 = Pipeline([('vect', vectorizer2), ('union', union)])
p3 = Pipeline([('ner', Nerer), ('vect', vectorizer), ('union', union2)])
p4 = Pipeline([('doc2ve', Doc2VecVectorizer()), ('union', union)])
scaler = MinMaxScaler()
ct = ColumnTransformer([('statement_lemma', p0, 'statement_spacy'), ('encoder_party', one_hot, ['party']),
                        ('subject_vect', p1, 'subject'), ('speaker_ner', p3, 'speaker_spacy'),
                        ('encoder_speaker', one_hot, ['speaker']), ('context_vect', p1, 'context'),
                        ('statement_features', make_pipeline(TextFeatures, scaler), 'statement'),
                        ('speaker_job_ner', p3, 'speaker_job_spacy')])
selector = SelectPercentile(percentile=80)
p = Pipeline([('column_transformer', ct), ('feature_selection', selector), ('logistic', logistic)])

p.fit(df_X, y)
cv = StratifiedKFold(10)
roc_auc = cross_val_score(p, df_X, y, scoring = "roc_auc", cv = cv)
print(np.mean(roc_auc))

wyniki.append(np.mean(roc_auc))

final_test = final_test.fillna(" ")
final_test["statement_spacy"] = list(nlp.pipe(final_test.statement))
final_test["speaker_job_spacy"] = list(nlp.pipe(final_test.speaker_job))
final_test["speaker_spacy"] = list(nlp.pipe(final_test.speaker))
scores = p.predict_proba(final_test)[:, 1]

with open('model.html', 'w', encoding = 'utf-8') as f:
    f.write(estimator_html_repr(p))

# not used variables
# ('context_doc2vec', p4, 'context'),

# from sklearn.linear_model import PassiveAggressiveClassifier
# pac = PassiveAggressiveClassifier()
# pa = Pipeline([('column_transformer', ct), ('feature_selection', selector), ('passive_aggressive', pac)])

# p.fit(df_X, y)
# cv = StratifiedKFold(3)
# roc_auc = cross_val_score(p, df_X, y, scoring = "roc_auc", cv = cv)
# print(np.mean(roc_auc))

a = [e.ents for e in df['statement_spacy']]

df['statement_ner'] = np.repeat(0, 10268)
i = 0
for x in a:
    df['statement_ner'][i] = ' '.join([y.label_ for y in x])
    i += 1

results = pd.DataFrame(scores)
results.columns = ['Anna Herud']

np.savetxt(r'FactChecking\results.txt', results.values)
# tfidft transformer na tfidf vectorizer
# bert
# łączenie modeli, votingclassifier
# izomapy (nieliniowe komponenty główne)