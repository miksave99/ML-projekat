import pandas as pd #uvoz biblioteke za rad sa podacima
import matplotlib.pyplot as plt # uvoz biblioteke za crtanje
import seaborn as sns #uvoz biblioteke za dijagrame
from sklearn.model_selection import train_test_split  #uvoz biblioteke za train i test skup
from sklearn.linear_model import LogisticRegression #uvoz biblioteke za Logističku regresiju
from sklearn.svm import SVC #uvoz biblioteke za SVC klasifikaciju
from sklearn.ensemble import RandomForestClassifier #uvoz biblioteke za Slučajna suma klasifikaciju
from sklearn.neighbors import KNeighborsClassifier #uvoz biblioteke za K-najbližih suseda
from sklearn.naive_bayes import GaussianNB #uvoz biblioteke za Naivni Bayes
from sklearn.metrics import auc, accuracy_score, classification_report, confusion_matrix, roc_curve #uvoz biblioteke za izračunavanje tačnosti klasifikacije
from sklearn.preprocessing import LabelEncoder #uvoz biblioteke za kodovanje reci
import pandas as pd #uvoz biblioteke za rad sa podacima
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np #uvoz biblioteke za rad sa nizovima, matricama ...




# Učitavanje podataka iz CSV fajla
data = pd.read_csv("tweet_emotions3_bert.csv", header=None, names=["sentiment","content","polarity"], encoding="ISO-8859-1")

# Izdvajanje podataka i oznaka
messages = data["content"]
labels = data["polarity"]

# Tokenizacija i vektorizacija teksta sa BERT
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Binary classification

encoded_texts = tokenizer.batch_encode_plus(
    messages.tolist(),
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
embeddings = model.bert(**encoded_texts).last_hidden_state

num_samples = embeddings.shape[0]
num_tokens = embeddings.shape[1]
embedding_size = embeddings.shape[2]
reshaped_embeddings = embeddings.view(num_samples, -1, embedding_size)
bert_features = reshaped_embeddings.detach().numpy()

# Deljenje podataka na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(bert_features, labels, test_size=0.2, random_state=42)

# Algoritmi
classifiers = {
    "1": LogisticRegression(),
    "2": SVC(),
    "3": RandomForestClassifier(),
    "4": KNeighborsClassifier(),
    "5": GaussianNB()
}

# Odabir algoritma
algorithm = input("Unesite željeni algoritam (1 - Logisticka Regresija, 2 - SVM, 3 - Slucajna suma, 4 - KNN, 5 - Naivni Bayes): ")
classifier = classifiers.get(algorithm)

if classifier is None:
    print("Nepodržani algoritam.")
else:
    # Treniranje algoritma
    classifier.fit(np.sum(X_train, axis=1), y_train)

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
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(8, 6))
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='Klasa {0} (AUC = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title("ROC kriva")
        plt.xlabel("Stopa lažno pozitivnih")
        plt.ylabel("Stopa istinito pozitivnih")
        plt.legend()
        plt.show()

    # Korisnički interfejs
    def user_interface():
        sentence = input("Unesite rečenicu: ")

        # Tokenizacija i vektorizacija unete rečenice sa BERT
        encoded_sentence = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        sentence_embedding = model.bert(**encoded_sentence).last_hidden_state
        sentence_embedding = sentence_embedding.view(1, -1, embedding_size)
        sentence_feature = sentence_embedding.detach().numpy()

        # Predviđanje oznake korišćenjem odabranog algoritma
        predicted_label = classifier.predict(np.sum(sentence_feature, axis=1))[0]

        print(f"Uneta rečenica je: {predicted_label}")

        # Izveštaj za trening skup podataka
        train_predictions = classifier.predict(np.sum(X_train, axis=1))
        train_accuracy = accuracy_score(y_train, train_predictions)
        train_report = classification_report(y_train, train_predictions, zero_division=1, output_dict=True)

        print("Izveštaj za trening skup podataka:")
        print("Tačnost:", train_accuracy)
        print("Preciznost:", train_report['macro avg']['precision'])
        print("Odziv:", train_report['macro avg']['recall'])
        print("F-mera:", train_report['macro avg']['f1-score'])

        # Izveštaj za test skup podataka
        test_predictions = classifier.predict(np.sum(X_test, axis=1))
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

        predicted_probabilities = classifier.predict_proba(np.sum(X_test, axis=1))
        plot_roc_curve_multi_class(y_test_encoded, predicted_probabilities)
        plot_confusion_matrix(y_test, test_predictions, "Matrica konfuzije za test skup podataka")

    user_interface()
