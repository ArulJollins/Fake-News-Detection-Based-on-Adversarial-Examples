import os
import re
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from scipy.spatial.distance import cosine
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Configuration
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DENSE_UNITS = 32
MODEL_DIR = "poisson_model"

class FakeNewsAdversarialDetector:
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
        self.model = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.scaler = None
        self.poisson_lambda = None
        self.intermediate_model = None  # to avoid retracing

    def clean_text(self, text):
        if pd.isnull(text):
            return ""
        text = text.lower()
        text = ''.join([c for c in text if c.isalnum() or c.isspace()])
        boilerplate = ["click here", "read more", "subscribe", "follow us", "share this"]
        for phrase in boilerplate:
            text = text.replace(phrase, "")
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        words = [self.lemmatizer.lemmatize(w) for w in words]
        return ' '.join(words)

    def load_data(self, true_path, fake_path):
        true_news = pd.read_csv(true_path).dropna(subset=["title", "text"])
        fake_news = pd.read_csv(fake_path).dropna(subset=["title", "text"])
        true_news["label"] = 0
        fake_news["label"] = 1
        data = pd.concat([true_news, fake_news], ignore_index=True)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        data["content"] = (data["title"] + " " + data["text"]).apply(self.clean_text)
        data = data[data["content"].str.split().apply(len) > 20]
        self.tokenizer.fit_on_texts(data["content"].values)
        sequences = self.tokenizer.texts_to_sequences(data["content"].values)
        X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        y = data["label"].values
        return X, y

    def build_model(self):
        inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
        x = Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM)(inputs)
        x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=False))(x)
        x = Dropout(0.5)(x)
        x = Dense(DENSE_UNITS, activation='relu', name='last_hidden')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, class_weights=None, epochs=5):
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=epochs, batch_size=32, class_weight=class_weights)

    def save_model(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model.save(os.path.join(MODEL_DIR, "model.h5"))
        with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
            pickle.dump(self.tokenizer, f)

    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model.h5"))
            with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
                self.tokenizer = pickle.load(f)
            print("✅ Model loaded")
        except:
            print("⚠️ No saved model found.")

    def get_activations(self, X):
        if self.intermediate_model is None:
            self.intermediate_model = Model(inputs=self.model.input,
                                            outputs=self.model.get_layer("last_hidden").output)
        return self.intermediate_model.predict(X)

    def build_poisson_distribution(self, X, y):
        real_activations = self.get_activations(X[y == 0])
        self.poisson_lambda = real_activations.mean(axis=0)
        self.scaler = MinMaxScaler()
        self.scaler.fit(real_activations)

    def is_adversarial(self, text, threshold=0.6):
        cleaned = self.clean_text(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
        activation = self.get_activations(padded)[0]
        norm_activation = self.scaler.transform([activation])[0]
        norm_expected = self.scaler.transform([self.poisson_lambda])[0]
        similarity = 1 - cosine(norm_activation, norm_expected)
        return similarity < threshold, similarity

    def predict_with_explanation(self, text):
        is_adv, similarity = self.is_adversarial(text)
        cleaned = self.clean_text(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
        prediction = self.model.predict(padded)[0][0]
        label = "fake" if prediction >= 0.6 else "real"
        explanation = f"Prediction: {label} ({prediction:.2f})\n"
        explanation += "⚠️ Adversarial Example Detected!\n" if is_adv else "✅ Normal Sample Detected.\n"
        explanation += f"Cosine similarity to expected: {similarity:.2f}"
        return {"prediction": label, "confidence": prediction, "explanation": explanation}


def main():
    true_path = "D:\\project\\project\\True.csv"
    fake_path = "D:\\project\\project\\Fake.csv"
    detector = FakeNewsAdversarialDetector()
    detector.load_model()

    if detector.model is None:
        X, y = detector.load_data(true_path, fake_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        detector.build_model()
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(enumerate(weights))
        detector.train(X_train, y_train, X_test, y_test, class_weights)
        detector.save_model()
        detector.build_poisson_distribution(X_train, y_train)
    else:
        X, y = detector.load_data(true_path, fake_path)
        detector.build_poisson_distribution(X, y)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = (detector.model.predict(X_test) > 0.6).astype(int)
    print(classification_report(y_test, y_pred, target_names=["real", "fake"]))

    while True:
        text = input("\nEnter news text (or leave empty to exit): ")
        if not text:
            break
        result = detector.predict_with_explanation(text)
        print(result["explanation"])


if __name__ == '__main__':
    main()
