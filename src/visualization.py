import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from . import config


def plot_class_distribution(data, title: str, save_as: str = None) -> None:
    """Affiche (et sauvegarde en option) la distribution des classes."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=config.LABEL_COLUMN, data=data)
    plt.title(title)
    _save_or_show(save_as)


def plot_word_cloud(data, label_value: str, title: str, save_as: str = None) -> None:
    """Génère un nuage de mots pour les emails d'une classe donnée."""
    subset = data[data[config.LABEL_COLUMN] == label_value]
    corpus = " ".join(subset[config.TEXT_COLUMN])

    wc = WordCloud(
        background_color="black", max_words=100, width=800, height=400
    ).generate(corpus)

    plt.figure(figsize=(7, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(title, fontsize=15)
    plt.axis("off")
    _save_or_show(save_as)


def plot_training_history(history, save_as: str = None) -> None:
    """Affiche l'évolution de l'accuracy et de la loss pendant l'entraînement."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))


    axes[0].plot(history.history["accuracy"], label="Entraînement")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Précision du modèle")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Époque")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Entraînement")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Perte du modèle")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Époque")
    axes[1].legend()

    plt.tight_layout()
    _save_or_show(save_as)



def plot_confusion_matrix(y_true, y_pred, save_as: str = None) -> None:
    """Affiche la matrice de confusion des prédictions."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
    disp.plot(cmap="Blues")
    plt.title("Matrice de confusion")
    _save_or_show(save_as)


def _save_or_show(save_as: str = None) -> None:
    """Sauvegarde la figure si un chemin est fourni, sinon l'affiche."""
    if save_as:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(config.OUTPUT_DIR, save_as), bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()
