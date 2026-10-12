# Email Spam Classifier

Classification of emails as **Spam / Ham** using a recurrent neural network (LSTM) with TensorFlow/Keras.

## Project Overview

This project implements a complete NLP pipeline for detecting unwanted emails (spam):

1. **Data exploration** — class distribution analysis
2. **Class balancing** — undersampling of the majority class (ham)
3. **Text cleaning** — removal of punctuation and stopwords
4. **Visualization** — word clouds for each class
5. **Tokenization & padding** of text sequences
6. **LSTM model** trained for binary classification
7. **Evaluation** — accuracy, confusion matrix, and classification report

## Project Structure

```text
email-spam-classifier/
├── data/
│   └── Emails.csv              # Dataset (not versioned, add it yourself)
├── outputs/                     # Generated charts and saved model
├── src/
│   ├── __init__.py
│   ├── config.py               # Centralized parameters
│   ├── data_preprocessing.py   # Data loading and text preprocessing
│   ├── visualization.py        # Charts (distribution, word clouds, curves)
│   ├── model.py                # LSTM model architecture
│   └── main.py                 # Main pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/email-spam-classifier.git
cd email-spam-classifier
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Place your `Emails.csv` file in the `data/` directory. The file must contain at least two columns:

| text                         | label |
| ---------------------------- | ----- |
| "Win a free prize now!!!"    | spam  |
| "Meeting rescheduled to 3pm" | ham   |

## Usage

From the project root directory:

```bash
python -m src.main
```

The script will:

* Display the progress in the console
* Save all charts to the `outputs/` directory
* Save the trained model to `outputs/spam_classifier_model.keras`

## Model Architecture

```text
Embedding(vocab_size, 32)
→ LSTM(16)
→ Dense(32, activation='relu')
→ Dropout(0.3)
→ Dense(1, activation='sigmoid')
```

**Optimizer:** Adam
**Loss:** Binary Crossentropy
**Callbacks:** EarlyStopping + ReduceLROnPlateau

## Results

After training, the script displays:

* Accuracy and loss on the test set
* A detailed classification report (precision, recall, F1-score)
* The confusion matrix
* Learning curves (accuracy/loss per epoch)

## Technologies Used

* Python 3.10+
* TensorFlow / Keras
* scikit-learn
* pandas, numpy
* matplotlib, seaborn
* nltk, wordcloud

## Author

Developed by **DJOUBER Dina**.

This project was developed to gain hands-on experience with an end-to-end text classification pipeline, including dataset balancing, text preprocessing, tokenization, LSTM modeling, and practical debugging.

