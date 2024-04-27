import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsRegressor

nba = pd.read_csv('nba_playoffs_meta.csv')

numerical_columns = nba.select_dtypes(include=[np.number]).columns
data_numeric = nba[numerical_columns]

data_numeric.dropna(inplace=True)

X = data_numeric.drop(columns=['game_won'])
y = data_numeric['game_won']

kbest = SelectKBest(score_func=f_regression, k='all')
kbest.fit(X, y)

selected_features = pd.DataFrame({'Feature': X.columns, 'Score': kbest.scores_})
selected_features_sorted = selected_features.sort_values(by='Score', ascending=False)

selected_features_sorted

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model1 = RandomForestClassifier(random_state=42)
model1.fit(X_train, y_train)
train_predictions = model1.predict(X_train)
test_predictions = model1.predict(X_test)
train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='r2')
cv_rmse_scores = -cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)



