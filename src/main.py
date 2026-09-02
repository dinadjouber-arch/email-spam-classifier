""""
Utilisation :
    python -m src.main
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report

from . import config
from . import data_preprocessing as dp
from . import visualization as viz
from .model import build_model


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    #  Chargement des données
    print("Chargement des données...")
    data = dp.load_data()
    print(f"Dimensions du dataset : {data.shape}")

    viz.plot_class_distribution(
        data, "Distribution initiale des classes", save_as="distribution_initiale.png"
    )

    # Equilibrage des classes
    print("Équilibrage des classes...")
    balanced_data = dp.balance_dataset(data)
    viz.plot_class_distribution(
        balanced_data, "Distribution équilibrée (Spam / Ham)", save_as="distribution_equilibree.png"
    )

    #  Nettoyage du texte
    print("Nettoyage du texte (ponctuation, stopwords)...")
    balanced_data = dp.clean_text_column(balanced_data)

    #  Visualisation (wordclouds)
    print("Génération des nuages de mots...")
    viz.plot_word_cloud(
        balanced_data, config.HAM_LABEL, "Emails non-Spam", save_as="wordcloud_ham.png"
    )
    viz.plot_word_cloud(
        balanced_data, config.SPAM_LABEL, "Emails Spam", save_as="wordcloud_spam.png"
    )

    #  Split train/test
    print("Séparation entraînement / test...")
    train_x, test_x, train_y, test_y = train_test_split(
        balanced_data[config.TEXT_COLUMN],
        balanced_data[config.LABEL_COLUMN],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=balanced_data[config.LABEL_COLUMN],
    )

    #  Tokenisation et padding
    print("Tokenisation...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_x)

    train_sequences = pad_sequences(
        tokenizer.texts_to_sequences(train_x),
        maxlen=config.MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )
    test_sequences = pad_sequences(
        tokenizer.texts_to_sequences(test_x),
        maxlen=config.MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    train_y = (train_y == config.SPAM_LABEL).astype(int).to_numpy()
    test_y = (test_y == config.SPAM_LABEL).astype(int).to_numpy()

    #Construction du modèle
    print("Construction du modèle...")
    vocab_size = len(tokenizer.word_index) + 1
    model = build_model(vocab_size)
    model.summary()

    # 8. Entraînement
    print("Entraînement du modèle...")
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        monitor="val_loss",
        restore_best_weights=True,
    )
    reduce_lr = ReduceLROnPlateau(
        patience=config.REDUCE_LR_PATIENCE,
        monitor="val_loss",
        factor=config.REDUCE_LR_FACTOR,
        verbose=0,
    )

    history = model.fit(
        train_sequences,
        train_y,
        validation_data=(test_sequences, test_y),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[reduce_lr, early_stopping],
    )

    #  Évaluation
    print("Évaluation du modèle...")
    test_loss, test_accuracy = model.evaluate(test_sequences, test_y)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")

    y_pred_proba = model.predict(test_sequences)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    print("\nRapport de classification :")
    print(classification_report(test_y, y_pred, target_names=["Ham", "Spam"]))

    viz.plot_confusion_matrix(test_y, y_pred, save_as="matrice_confusion.png")
    viz.plot_training_history(history, save_as="courbes_entrainement.png")

    #Sauvegarde du modèle
    model.save(config.MODEL_PATH)
    with open(config.TOKENIZER_PATH, "w", encoding="utf-8") as f:
       f.write(tokenizer.to_json())
    print(f"\nModèle sauvegardé dans     : {config.MODEL_PATH}")
    print(f"Tokenizer sauvegardé dans  : {config.TOKENIZER_PATH}")
    print("\nTu peux maintenant tester le modèle sur tes propres messages avec :")
    print("    python -m src.predict")


if __name__ == "__main__":
    main()
