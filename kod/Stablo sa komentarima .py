import pandas as pd #uvoz biblioteke za rad sa podacima
import matplotlib.pyplot as plt # uvoz biblioteke za crtanje
from sklearn.model_selection import train_test_split  #uvoz biblioteke za train i test skup
from sklearn.feature_extraction.text import CountVectorizer #pretvaranje teksta u matricu brojeva.
from sklearn.tree import DecisionTreeClassifier #uvoz biblioteke za klasifukator stabla odlučivanja
from sklearn.metrics import accuracy_score #uvoz biblioteke za ROC krivu
from sklearn import tree #uvoz biblioteke za analizu stabla


# Učitavanje podataka iz CSV fajla
data = pd.read_csv("tweet_emotions3.csv", header=None, names=["content","polarity"], encoding="ISO-8859-1")

# Uklanjanje redova sa NaN vrednostima
data = data.dropna()

# Izdvajanje podataka i oznaka
text = data["content"]
category = data["polarity"]

# Vektorizacija teksta
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text)

# Deljenje podataka na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X, category, test_size=0.2, random_state=42)

# Treniranje klasifikatora
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predviđanje na test skupu
y_pred = clf.predict(X_test)

# Tačnost klasifikacije
accuracy = accuracy_score(y_test, y_pred)
print("Tačnost:", accuracy)

# Vizualizacija stabla odlučivanja
fig, ax = plt.subplots(figsize=(8, 8))
tree.plot_tree(clf, filled=True, ax=ax, feature_names=list(vectorizer.vocabulary_.keys()), class_names=list(clf.classes_))
plt.show()