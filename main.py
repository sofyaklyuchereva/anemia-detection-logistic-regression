import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


#getting data 
df = pd.read_csv("file_.csv")
#shuffling data
df = df.sample(frac=1, random_state=42).reset_index(drop=True) 
X = df[['Sex','%Red Pixel', '%Green pixel', '%Blue pixel', 'Hb']]
y = df[['Anaemic']]

print(df)
#encoding labels
le_anaemic = LabelEncoder()
y_encoded = le_anaemic.fit_transform(y)
le_sex = LabelEncoder()
X['Sex'] = le_sex.fit_transform(X['Sex'])

#scaling the features
scaler = StandardScaler()
X[['%Red Pixel', '%Green pixel', '%Blue pixel', 'Hb']] = scaler.fit_transform(X[['%Red Pixel', '%Green pixel', '%Blue pixel', 'Hb']])


#splitting the test and train data
X_train, X_test, y_train, y_test = train_test_split(X,y_encoded, test_size=0.3, random_state=42)

#training the model
logreg = LogisticRegression(C=0.01, random_state=16)
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

#confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)

#evaluation metrics
target = le_anaemic.inverse_transform([0,1])
print(metrics.classification_report(y_test, y_pred, target_names = target))

#ROC curve
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()


