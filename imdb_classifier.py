import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

print("=" * 80)
print("KLASYFIKATOR RECENZJI IMDB - ANALIZA SENTYMENTU")
print("=" * 80)

# ============================================================================
# WCZYTANIE DANYCH
# ============================================================================
print("\n[KROK 1] Wczytywanie danych IMDB...")

# UWAGA: Zmień ścieżkę na swoją
df = pd.read_csv('IMDB Dataset.csv')

print(f"✓ Wczytano {len(df)} recenzji")
print(f"\nRozkład sentymentu:")
print(df['sentiment'].value_counts())

# Konwersja na binarne (0 = negative, 1 = positive)
df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})

# Ze względu na wielkość datasetu, weź próbkę (opcjonalnie)
# df = df.sample(n=10000, random_state=42)  # Odkomentuj dla szybszego działania

# ============================================================================
# PREPROCESSING
# ============================================================================
print("\n[KROK 2] Preprocessing...")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # Usuń tagi HTML
    text = re.sub(r'<.*?>', '', text)
    # Małe litery
    text = text.lower()
    # Usuń znaki specjalne
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenizacja
    tokens = word_tokenize(text)
    # Stopwords i lematyzacja
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return ' '.join(tokens)


print("Przetwarzanie recenzji (może potrwać kilka minut)...")
df['processed'] = df['review'].apply(preprocess_text)

print("✓ Preprocessing zakończony")

# ============================================================================
# PODZIAŁ DANYCH I WEKTORYZACJA
# ============================================================================
print("\n[KROK 3] Podział danych i wektoryzacja...")

X = df['processed']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"✓ Zbiór treningowy: {X_train_tfidf.shape}")
print(f"✓ Zbiór testowy: {X_test_tfidf.shape}")

# ============================================================================
# TRENING MODELI
# ============================================================================
print("\n[KROK 4] Trening modeli...")

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Linear SVM': LinearSVC(random_state=42, max_iter=2000),
}

results = {}

for name, model in models.items():
    print(f"\nTrenowanie: {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {'accuracy': acc, 'f1': f1, 'y_pred': y_pred}

    print(f"✓ Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

# ============================================================================
# WYNIKI
# ============================================================================
print("\n" + "=" * 80)
print("WYNIKI KOŃCOWE")
print("=" * 80)

for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            metrics['y_pred'],
            target_names=['Negative', 'Positive'],
            digits=4,
        )
    )

# Wizualizacja
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, metrics) in enumerate(results.items()):
    cm = confusion_matrix(y_test, metrics['y_pred'])
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Neg', 'Pos'],
        yticklabels=['Neg', 'Pos'],
        ax=axes[idx],
    )
    axes[idx].set_title(f'{name}\nAcc: {metrics["accuracy"]:.3f}')

plt.tight_layout()
plt.savefig('imdb_results.png', dpi=300)
print("\n✓ Wykres zapisany jako 'imdb_results.png'")
plt.show()

print("\n" + "=" * 80)
print("✓ ANALIZA ZAKOŃCZONA!")
print("=" * 80)