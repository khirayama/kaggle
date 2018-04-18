import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv('./input/train.csv')
df_train = df_train.replace('male', 0).replace('female', 1)
df_train['Age'].fillna(df_train.Age.median(), inplace = True)
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
train_data = df_train.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 1).values

xs = train_data[:, 2:]
y = train_data[:, 1]

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(xs, y)

test_df = pd.read_csv('./input/test.csv')
test_df = test_df.replace('male', 0).replace('female', 1)
test_df['Age'].fillna(df_train.Age.median(), inplace = True)
test_df['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
test_data = test_df.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 1).values

xs_test = test_data[:, 1:]
output = forest.predict(xs_test)

zip_data = zip(test_data[:,0].astype(int), output.astype(int))
predict_data = list(zip_data)

with open('predict_result_data.csv', 'w') as f:
    writer = csv.writer(f, lineterminator = '\n')
    writer.writerow(['PassengerId', 'Survived'])
    for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])
