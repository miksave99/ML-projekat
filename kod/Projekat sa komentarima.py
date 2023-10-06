import pandas as pd #uvoz biblioteke za rad sa podacima
import numpy as np #uvoz biblioteke za rad sa nizovima, matricama ...
import seaborn as sns #uvoz biblioteke za dijagrame
import matplotlib.pyplot as plt # uvoz biblioteke za crtanje
from sklearn.feature_extraction.text import CountVectorizer #pretvaranje teksta u matricu brojeva.
from sklearn.model_selection import train_test_split #uvoz biblioteke za train i test skup
from sklearn.linear_model import LogisticRegression #uvoz biblioteke za Logističku regresiju
from sklearn.ensemble import RandomForestClassifier #uvoz biblioteke za Slučajna suma klasifikaciju
from sklearn.metrics import auc #uvoz biblioteke za ROC krivu
from sklearn.svm import SVC #uvoz biblioteke za SVC klasifikaciju
from sklearn.preprocessing import LabelEncoder #uvoz biblioteke za kodovanje reci
from sklearn.naive_bayes import MultinomialNB #uvoz biblioteke za Naivni Bayes
from sklearn.neighbors import KNeighborsClassifier #uvoz biblioteke za K-najbližih suseda
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve #uvoz biblioteke za izračunavanje tačnosti klasifikacije

# Učitavanje podataka iz CSV fajla
data = pd.read_csv("tweet_emotions3.csv", header=None, names=["sentiment","content","polarity"], encoding="ISO-8859-1")
print(data.head())
# Uklanjanje redova sa NaN vrednostima
data = data.dropna()

# Izdvajanje podataka i oznak0a
text = data["content"]
category = data["polarity"]

# Vektorizacija teksta
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(text)

# Deljenje podataka na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(features, category, test_size=0.2, random_state=42)

# Algoritam 1: Logistička regresija
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Algoritam 2: Slučajna suma
random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=10)
random_forest_model.fit(X_train, y_train)

# Algoritam 3: SVM
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Algoritam 4: Naivni Bayes
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)

# Algoritam 5: K-najbližih suseda
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)


# Funkcija za prikaz matrice konfuzije
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    classes = np.unique(np.concatenate((y_true, y_pred)))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predviđene vrednosti")
    plt.ylabel("Stvarne vrednosti")
    plt.show()

# Funkcija za prikaz ROC krive
def plot_roc_curve_multi_class(y_true, y_scores):
    n_classes = y_scores.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    class_labels = ["positive", "neutral", "negative"]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        linestyle = "-" if i == 0 else "--"
        plt.plot(fpr[i], tpr[i], label="ROC kriva za klasu {} ".format(class_labels[i], roc_auc[i]), linestyle=linestyle)

    plt.plot([0, 1], [0, 1], linestyle="--", color="r", label="Slučajna klasifikacija")
    plt.title("ROC kriva")
    plt.xlabel("Stopa lažno pozitivnih")
    plt.ylabel("Stopa istinito pozitivnih")
    plt.legend()
    plt.show()



# Funkcija za odabir algoritma
def choose_algorithm():
    print("Odaberite algoritam za klasifikaciju:")
    print("1 - Logistička regresija")
    print("2 - Slučajna suma")
    print("3 - SVM")
    print("4 - Naivni Bayes")
    print("5 - K-najbližih suseda")
    choice = input("Unesite broj odabranog algoritma: ")
    return int(choice)


# Korisnički interfejs
def user_interface():
    choice = choose_algorithm()

    if choice == 1:
        model = logistic_model
        model_name = "Logistička regresija"
    elif choice == 2:
        model = random_forest_model
        model_name = "Slučajna suma"
    elif choice == 3:
        model = svm_model
        model_name = "SVM"
    elif choice == 4:
        model = naive_bayes_model
        model_name = "Naivni Bayes"
    elif choice == 5:
        model = knn_model
        model_name = "K-najbližih suseda"

    sentence = input("Unesite rečenicu: ")
    sentence_features = vectorizer.transform([sentence])
    result = model.predict(sentence_features)[0]

    print(f"Uneta rečenica je: {result}")

    # Dodajte proveru broja klasa
    num_classes = len(np.unique(y_test))
    if num_classes > 4:
        print("Nekorektno podešene klase. Očekuje se binarna ili višeklasna klasifikacija.")
        return

    # Izveštaj za trening skup podataka
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_report = classification_report(y_train, train_predictions, zero_division=1, output_dict=True)

    print("Izveštaj za trening skup podataka:")
    print("Tačnost:", train_accuracy)
    print("Preciznost:", train_report['macro avg']['precision'])
    print("Odziv:", train_report['macro avg']['recall'])
    print("F-mera:", train_report['macro avg']['f1-score'])

    # Izveštaj za test skup podataka
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_report = classification_report(y_test, test_predictions, zero_division=1, output_dict=True)

    print("Izveštaj za test skup podataka:")
    print("Tačnost:", test_accuracy)
    print("Preciznost:", test_report['macro avg']['precision'])
    print("Odziv:", test_report['macro avg']['recall'])
    print("F-mera:", test_report['macro avg']['f1-score'])

    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    y_test_encoded = np.eye(len(label_encoder.classes_))[y_test_encoded]

    predicted_probabilities = model.predict_proba(X_test)
    plot_roc_curve_multi_class(y_test_encoded, predicted_probabilities)

    plot_confusion_matrix(y_test, test_predictions, "Matrica konfuzije za test skup podataka")

    train_predictions = model.predict(X_train)
    plot_confusion_matrix(y_train, train_predictions, "Matrica konfuzije za trening skup podataka")


# Pokretanje korisničkog interfejsa
user_interface()
