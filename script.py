import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class CabinValuizer(BaseEstimator, TransformerMixin):
    def __init__(self, simple):
        self.simple = simple
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        for i in range(len(X)):
            if type(X[i][0]) == float:
                X[i][0] = 'Z'

        place = np.zeros(len(X))
        for i in range(len(X)):
            if 'Z' in X[i][0]:
                continue
            if 'A' in X[i][0]:
                place[i] = 11
            if 'B' in X[i][0]:
                place[i] = 12
            if 'C' in X[i][0]:
                place[i] = 13
            if 'D' in X[i][0]:
                place[i] = 14
            if 'E' in X[i][0]:
                place[i] = 15
            if 'F' in X[i][0]:
                place[i] = 16
            if 'G' in X[i][0]:
                place[i] = 17

        X = place.reshape(len(place),1)

        return X

class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

train = pd.read_csv("/home/guillaume/Projects/DataAnalytics/Titanic/train.csv")
test = pd.read_csv("/home/guillaume/Projects/DataAnalytics/Titanic/test.csv")

encodeAtt = ["Sex"] #Binary encode sex(male/female)
changeAtt = ["Cabin"] #Attributes to be manually changed (place) Cabin to be changed
numAtt = ["Age", "Fare", "Pclass", "SibSp", "Parch"] #Attributes to be imputed and scaled, etc, etc
y_labels = ["Survived"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(numAtt)),
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

enc_pipeline = Pipeline([
    ('selector', DataFrameSelector(encodeAtt)),
    ('label_binarizer', MyLabelBinarizer()),
])

chg_pipeline = Pipeline([
    ('selector', DataFrameSelector(changeAtt)),
    ('changer', CabinValuizer(False)),
    ('std_scaler', StandardScaler()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("enc_pipeline", enc_pipeline),
    ("chg_pipeline", chg_pipeline),
])

num_train_prepared = num_pipeline.fit_transform(train)
enc_train_prepared = enc_pipeline.fit_transform(train)
chg_train_prepared = chg_pipeline.fit_transform(train)

X_train = full_pipeline.fit_transform(train)
y_train = y_train = (train.Survived.values).reshape(len(train.Survived.values),1)
X_test = full_pipeline.fit_transform(test)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
knn_clf = KNeighborsClassifier()
dct_clf = DecisionTreeClassifier()
nbs_clf = GaussianNB()
svc_clf = SVC(gamma='scale')
log_clf = LogisticRegression(solver='lbfgs')
rnd_clf = RandomForestClassifier(n_estimators=100)

vot_clf = VotingClassifier(
    estimators=[('lr', log_clf),
                ('kn', knn_clf),
                ('sv', svc_clf),
                ('rf', rnd_clf)]
)

for clf in (sgd_clf, knn_clf, dct_clf, nbs_clf, svc_clf, log_clf, rnd_clf, vot_clf):
    clf.fit(X_train, y_train.ravel())
    y_pred = cross_val_score(clf, X_train, y_train.ravel(), scoring='accuracy', cv=10)
    print(clf.__class__.__name__, y_pred.mean())



# #Take only Log, Rnd, Knn, SVC and Voting
log_pred = log_clf.predict(X_test)
log_sumb = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":log_pred})
log_sumb.to_csv("log_sumb.csv", index=False)

knn_pred = knn_clf.predict(X_test)
knn_sumb = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":knn_pred})
knn_sumb.to_csv("knn_sumb.csv", index=False)

svc_pred = svc_clf.predict(X_test)
svc_sumb = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":svc_pred})
svc_sumb.to_csv("svc_sumb.csv", index=False)

rnd_pred = rnd_clf.predict(X_test)
rnd_sumb = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":rnd_pred})
rnd_sumb.to_csv("rnd_sumb.csv", index=False)

vot_pred = vot_clf.predict(X_test)
vot_sumb = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":vot_pred})
vot_sumb.to_csv("vot_sumb.csv", index=False)

print("Done writing.")

svc_param = {
    'C':[1,10,100,1000],
    'gamma':[1,0.1,0.001,0.0001],
    'kernel':['linear','rbf'],
    'degree':[3,4]
}

svc_grid = GridSearchCV(svc_clf, svc_param, cv=10)

svc_grid.fit(X_train, y_train)

