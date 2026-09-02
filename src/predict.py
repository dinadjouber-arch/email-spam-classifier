import json
import warnings
 
warnings.filterwarnings("ignore")
 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
from . import config
from .data_preprocessing import remove_punctuation, remove_stopwords, download_nltk_resources
from nltk.corpus import stopwords
 
 
def load_trained_model():
    """Charge le modèle et le tokenizer sauvegardés lors de l'entrainement."""
    try:
        model = tf.keras.models.load_model(config.MODEL_PATH)
    except (IOError, OSError) as exc:
        raise FileNotFoundError(
            f"Modèle introuvable à {config.MODEL_PATH}. "
            f"Lance d'abord 'python -m src.main' pour entrainer et sauvegarder un modèle."
        ) from exc
 
    try:
        with open(config.TOKENIZER_PATH, "r", encoding="utf-8") as f:
            tokenizer = tokenizer_from_json(f.read())
    except (IOError, OSError) as exc:
        raise FileNotFoundError(
            f"Tokenizer introuvable à {config.TOKENIZER_PATH}. "
            f"Lance d'abord 'python -m src.main' pour le générer."
        ) from exc
 
    return model, tokenizer
 
 
def preprocess_message(text: str, stop_words: set) -> str:
    """Applique exactement le même nettoyage que pendant l'entraînement."""
    text = text.replace("Subject", "")
    text = remove_punctuation(text)
    text = remove_stopwords(text, stop_words)
    return text
 
 
def predict_message(text: str, model, tokenizer, stop_words: set) -> tuple:
    """
    Prédit si un message est spam ou ham.
    Retourne (label prédit, probabilité de spam entre 0 et 1).
    """
    cleaned = preprocess_message(text, stop_words)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence, maxlen=config.MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
    )
 
    probability = float(model.predict(padded, verbose=0)[0][0])
    label = config.SPAM_LABEL if probability > 0.5 else config.HAM_LABEL
 
    return label, probability
 
 
def main():
    print("Chargement du modèle...")
    model, tokenizer = load_trained_model()
 
    download_nltk_resources()
    stop_words = set(stopwords.words("english"))
 
    print("\nModèle prêt. Tapez un message en anglais à classer (ou 'quit' pour sortir).\n")
 
    while True:
        user_input = input("Message > ").strip()
 
        if user_input.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break
 
        if not user_input:
            continue
 
        label, probability = predict_message(user_input, model, tokenizer, stop_words)
 
        if label == config.SPAM_LABEL:
            print(f"  -> SPAM  (confiance : {probability * 100:.1f}%)\n")
        else:
            print(f"  -> HAM   (confiance : {(1 - probability) * 100:.1f}%)\n")
 
 
if __name__ == "__main__":
    main()
 