# Email Spam Classifier

Classification d'emails **Spam / Ham** à l'aide d'un réseau de neurones récurrent (LSTM) avec TensorFlow/Keras.

## Aperçu du projet

Ce projet implémente un pipeline complet de NLP pour détecter les emails indésirables (spam) :

1. **Exploration des données** — distribution des classes
2. **Équilibrage** — sous-échantillonnage de la classe majoritaire (ham)
3. **Nettoyage du texte** — suppression de la ponctuation et des stopwords
4. **Visualisation** — nuages de mots (wordclouds) par classe
5. **Tokenisation & padding** des séquences
6. **Modèle LSTM** entraîné pour la classification binaire
7. **Évaluation** — accuracy, matrice de confusion, rapport de classification

##  Structure du projet

```
email-spam-classifier/
├── data/
│   └── Emails.csv          # Dataset (non versionné, à ajouter toi-même)
├── outputs/                # Graphiques et modèle sauvegardé (générés automatiquement)
├── src/
│   ├── __init__.py
│   ├── config.py            # Paramètres centralisés
│   ├── data_preprocessing.py # Chargement et nettoyage des données
│   ├── visualization.py      # Graphiques (distribution, wordclouds, courbes)
│   ├── model.py               # Architecture du modèle LSTM
│   └── main.py                 # Pipeline principal
├── requirements.txt
├── .gitignore
└── README.md
```

##  Installation

```bash
git clone https://github.com/<ton-utilisateur>/email-spam-classifier.git
cd email-spam-classifier
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Données

Place ton fichier `Emails.csv` dans le dossier `data/`. Le fichier doit contenir au minimum deux colonnes :

| text                          | label |
|--------------------------------|-------|
| "Win a free prize now!!!"      | spam  |
| "Meeting rescheduled to 3pm"   | ham   |

##  Utilisation

Depuis la racine du projet :

```bash
python -m src.main
```

Le script va :
- Afficher la progression dans la console
- Sauvegarder tous les graphiques dans `outputs/`
- Sauvegarder le modèle entraîné dans `outputs/spam_classifier_model.keras`

## Architecture du modèle

```
Embedding(vocab_size, 32)
→ LSTM(16)
→ Dense(32, activation='relu')
→ Dropout(0.3)
→ Dense(1, activation='sigmoid')
```

Optimiseur : Adam · Loss : Binary Crossentropy · Callbacks : EarlyStopping + ReduceLROnPlateau

## Résultats

Après entraînement, le script affiche :
- La précision (accuracy) et la perte (loss) sur l'ensemble de test
- Un rapport de classification détaillé (precision, recall, F1-score)
- La matrice de confusion
- Les courbes d'apprentissage (accuracy/loss par époque)

##  Technologies utilisées

- Python 3.10+
- TensorFlow / Keras
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- nltk, wordcloud

##  Auteur

Développé par DJOUBER Dina.

Ce projet a été réalisé pour apprendre concrètement comment fonctionne un pipeline de classification de texte de bout en bout : équilibrage du dataset, prétraitement du texte, tokenisation, modèle LSTM, et tout le débogage pratique qui va avec (configuration de l'environnement, un bug de padding/masking qui a silencieusement empêché l'entraînement de fonctionner, adaptation du code à un dataset avec un format de colonnes différent, etc.).