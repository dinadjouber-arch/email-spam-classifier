"""
Chargement et nettoyage des données d'emails.
"""

import string
import pandas as pd
import nltk
from nltk.corpus import stopwords

from . import config


def download_nltk_resources() -> None:
    """Télécharge les ressources NLTK nécessaires (silencieux si déjà présentes)."""
    nltk.download("stopwords", quiet=True)



def load_data(path: str = config.DATA_PATH) -> pd.DataFrame:
    """
    Charge le fichier CSV d'emails.

    Le fichier doit contenir au minimum les colonnes définies dans
    config.TEXT_COLUMN et config.LABEL_COLUMN.
    """
    try:
        data = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Fichier introuvable : {path}. "
            f"Place ton fichier Emails.csv dans le dossier 'data/'."
        ) from exc

    missing_cols = {config.TEXT_COLUMN, config.LABEL_COLUMN} - set(data.columns)
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans le CSV : {missing_cols}. "
            f"Colonnes trouvées : {list(data.columns)}"
        )

    return data


def balance_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Équilibre le dataset en sous-échantillonnant la classe majoritaire (ham)
    pour qu'elle ait le même nombre d'exemples que la classe minoritaire (spam).
    """
    ham_msg = data[data[config.LABEL_COLUMN] == config.HAM_LABEL]
    spam_msg = data[data[config.LABEL_COLUMN] == config.SPAM_LABEL]

    if len(spam_msg) == 0:
        raise ValueError("Aucun exemple 'spam' trouvé dans les données.")

    ham_balanced = ham_msg.sample(n=len(spam_msg), random_state=config.RANDOM_STATE)
    balanced = pd.concat([ham_balanced, spam_msg]).reset_index(drop=True)
    return balanced


def remove_punctuation(text: str) -> str:
    """Supprime toute la ponctuation d'une chaîne de caractères."""
    translation_table = str.maketrans("", "", string.punctuation)
    return text.translate(translation_table)



def remove_stopwords(text: str, stop_words: set) -> str:
    """Supprime les mots vides (stopwords) et met le texte en minuscules."""
    important_words = [
        word.lower() for word in str(text).split() if word.lower() not in stop_words
    ]
    return " ".join(important_words)


def clean_text_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le pipeline de nettoyage complet sur la colonne de texte :
    suppression du mot 'Subject', de la ponctuation, puis des stopwords.
    """
    download_nltk_resources()
    stop_words = set(stopwords.words("english"))

    data = data.copy()
    data[config.TEXT_COLUMN] = (
        data[config.TEXT_COLUMN]
        .astype(str)
        .str.replace("Subject", "", regex=False)
        .apply(remove_punctuation)
        .apply(lambda text: remove_stopwords(text, stop_words))
    )
    
    return data
