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

