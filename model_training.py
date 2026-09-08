import pandas as pd

df = pd.read_csv('final_spotify_audio_dataset_with_lyrics_and_mood.csv')

x = df[[ 'danceability', 'energy', 'tempo', 'acousticness', 'instrumentalness', 'loudness', 'speechiness']]

y = df['mood']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC





x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42 )

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
lr = LogisticRegression(max_iter = 1000, random_state = 42)
gb = GradientBoostingClassifier(n_estimators = 100, random_state = 42)
svm = SVC(random_state = 42)




tree_model = {'Random Forest' : rf, 'Gradient Boosting' : gb}

scaled_model = {'Logistic Regression' : lr, 'Support Vector Machine' : svm}


for name, model in tree_model.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"{name} - Accuracy: {model.score(x_test, y_test)}")


for name, model in scaled_model.items():
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    print(f'{name} - Accuracy: {model.score(x_test_scaled, y_test)}')

# rf.fit(x_train, y_train)

# y_pred = rf.predict(x_test)

# from sklearn.metrics import classification_report, confusion_matrix

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))