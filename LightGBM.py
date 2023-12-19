import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Drop unnecessary columns
titanic = titanic.drop(['deck', 'embark_town', 'alive'], axis=1)

# Replace missing values with the median or mode
titanic['age'] = titanic['age'].fillna(titanic['age'].median())
titanic['fare'] = titanic['fare'].fillna(titanic['fare'].mode()[0])
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])

# Convert categorical variables to numerical variables
titanic['sex'] = pd.Categorical(titanic['sex']).codes
titanic['embarked'] = pd.Categorical(titanic['embarked']).codes

# Split the dataset into input features and the target variable
X = titanic.drop('survived', axis=1)
y = titanic['survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class_dict = {
"Third": 3,
"First": 1,
"Second": 2
}
who_dict = {
"child": 0,
"woman": 1,
"man": 2
}
X_train['class'] = X_train['class'].apply(lambda x: class_dict[x])
X_train['who'] = X_train['who'].apply(lambda x: who_dict[x])
X_test['class'] = X_test['class'].apply(lambda x: class_dict[x])
X_test['who'] = X_test['who'].apply(lambda x: who_dict[x])

params = {
'objective': 'binary',
'boosting_type': 'gbdt',
'num_leaves': 31,
'learning_rate': 0.05,
'feature_fraction': 0.9
}
clf = lgb.LGBMClassifier(**params)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))

model = lgb.LGBMClassifier(num_leaves=31, min_data_in_leaf=20, max_depth=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

