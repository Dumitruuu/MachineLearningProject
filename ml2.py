from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn import tree
import pandas as pd
pd.options.mode.chained_assignment = None # no warnings
import numpy
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
X = df[['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'TotalCharges', 'MonthlyCharges', 'SeniorCitizen']]
print(X, X.describe())
X['Partner'] = X.Partner.map(dict(Yes=1, No=0))
X['Dependents'] = X.Dependents.map(dict(Yes=1, No=0))
X['PhoneService'] = X.PhoneService.map(dict(Yes=1, No=0))
X['PaperlessBilling'] = X.PaperlessBilling.map(dict(Yes=1, No=0))
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
X = X.fillna(0)
print(X)
y= df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
scor = balanced_accuracy_score(y_test, y_pred)
print(scor)

