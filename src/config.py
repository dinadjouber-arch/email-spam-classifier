
import os

# --- Chemins ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Emails.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "spam_classifier_model.keras")
TOKENIZER_PATH = os.path.join(OUTPUT_DIR, "tokenizer.json")

# --- Colonnes attendues dans le CSV ---
TEXT_COLUMN = "Message"
LABEL_COLUMN = "Category"
SPAM_LABEL = "spam"
HAM_LABEL = "ham"

# --- Pretraitement du texte ---
RANDOM_STATE = 42
MAX_SEQUENCE_LENGTH = 100

# --- Split train/test ---
TEST_SIZE = 0.2

# --- Modele ---
EMBEDDING_DIM = 32
LSTM_UNITS = 16
DENSE_UNITS = 32
DROPOUT_RATE = 0.3

# --- Entrainement ---
EPOCHS = 30
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5