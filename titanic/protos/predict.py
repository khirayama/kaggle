import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv('./input/train.csv')

df_train['Age'].fillna(df_train.Age.median(), inplace = True)

sex_dum = pd.get_dummies(df_train['Sex'])
df_train_proc = pd.concat((df_train, sex_dum), axis = 1)
df_train_proc = df_train_proc.drop('Sex', axis = 1)

emb_dum = pd.get_dummies(df_train['Embarked'])
df_train_proc = pd.concat((df_train_proc, emb_dum), axis = 1)
df_train_proc = df_train_proc.drop('Embarked', axis = 1)

train_data = df_train_proc.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis = 1).values

xs = train_data[:, 2:]
y = train_data[:, 1]

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(xs, y)

df_test = pd.read_csv('./input/test.csv')

df_test['Age'].fillna(df_train.Age.median(), inplace = True)

sex_dum = pd.get_dummies(df_test['Sex'])
df_test_proc = pd.concat((df_test, sex_dum), axis = 1)
df_test_proc = df_test_proc.drop('Sex', axis = 1)

emb_dum = pd.get_dummies(df_test['Embarked'])
df_test_proc = pd.concat((df_test_proc, emb_dum), axis = 1)
df_test_proc = df_test_proc.drop('Embarked', axis = 1)

test_data = df_test_proc.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis = 1).values

xs_test = test_data[:, 1:]
output = forest.predict(xs_test)

zip_data = zip(test_data[:,0].astype(int), output.astype(int))
predict_data = list(zip_data)

with open('predict_result_data.csv', 'w') as f:
    writer = csv.writer(f, lineterminator = '\n')
    writer.writerow(['PassengerId', 'Survived'])
    for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])
