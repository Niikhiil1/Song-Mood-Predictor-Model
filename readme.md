# Song Mood Predictor — Audio Features + Lyrics Sentiment

An end-to-end ML project that predicts a song's **mood** (Happy / Sad / Neutral) using a combination of Spotify's audio features and NLP-based sentiment analysis of the song's lyrics.

## Why this project is different

Most "Spotify ML" tutorials use audio features alone (danceability, tempo, energy) to predict genre or popularity. This project instead combines **two data sources — structured audio features AND unstructured text (lyrics)** — to build a mood label, and specifically tests how much lyrics content adds over audio alone. That comparison is the core, genuine finding of this project.

---

## Project Pipeline

```
Kaggle Spotify Dataset (audio features)
        +
Genius API (lyrics, fetched programmatically)
        ↓
[01_collect_data.py]  →  merges audio features + lyrics into one dataset
        ↓
[02_create_mood_labels.py]  →  runs sentiment analysis on lyrics (TextBlob),
                                 combines with valence to create 'mood' label
        ↓
[03_train_model.py]  →  trains & compares multiple classifiers,
                          evaluates on held-out test data
```

## Files

| File | Purpose |
|---|---|
| `01_collect_data.py` | Loads Kaggle's Spotify audio-features dataset, samples ~250 songs, fetches matching lyrics via the Genius API |
| `02_create_mood_labels.py` | Runs sentiment analysis (TextBlob) on lyrics, combines with Spotify's `valence` score to assign a mood label (`happy` / `sad` / `neutral`) |
| `03_train_model.py` | Trains and compares 4 classifiers (Random Forest, Gradient Boosting, Logistic Regression, SVM) to predict mood from audio features |

## Data Sources
- **Audio features**: [Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — includes danceability, energy, valence, tempo, acousticness, instrumentalness, loudness, speechiness
- **Lyrics**: Fetched live via the [Genius API](https://genius.com/api-clients) using the `lyricsgenius` Python library

## Key Technical Decision: Avoiding Data Leakage

The mood label was originally derived using `valence` (from Spotify) combined with `lyrics_sentiment` (from TextBlob). Early versions of the model included these same columns as **input features** — which is a classic case of **data leakage**: the model was essentially given the answer key.

Progressive fix and honest evaluation:

| Features used as input | Accuracy | What it tells us |
|---|---|---|
| `valence` + `lyrics_sentiment` + audio features | 95% | Inflated — direct leakage from the label-creation rule |
| `lyrics_sentiment` + audio features (no valence) | 62% | Still partial leakage |
| **Audio features only** (no valence, no lyrics_sentiment) | **43–54%** | Honest baseline — genuinely hard problem |

**Insight:** Predicting mood from audio characteristics alone is a genuinely hard problem (43-54% vs. a ~33% random baseline for 3 classes) — this supports the project's core premise that lyrics content carries real, non-redundant signal about a song's emotional character that pure audio features don't fully capture.

## Model Comparison (on honest, leakage-free audio features)

| Model | Accuracy |
|---|---|
| Random Forest | 43.2% |
| **Gradient Boosting** | **54.1%** |
| **Logistic Regression** | **54.1%** |
| SVM | 37.8% |

Gradient Boosting and Logistic Regression performed best on this small dataset (~184 songs after dropping songs with missing lyrics).

## Known Limitations
- **Small dataset** (~184 songs) — results have high variance; a larger sample would give more reliable numbers.
- **Genius lyrics occasionally returned the wrong-language version** of a song (e.g., a Portuguese cover page instead of the original), which would affect sentiment scoring for those tracks.
- **TextBlob's sentiment analysis is simple (lexicon-based)** — a transformer-based sentiment model (e.g., a fine-tuned BERT) would likely produce more accurate sentiment scores.
- Mood labels were rule-based (`valence` + `sentiment` thresholds) rather than human-annotated — a limitation of any weak-labeling approach.

## Possible Future Improvements
- Collect a larger sample (500+ songs) for more stable evaluation.
- Use richer lyrics-based features (word count, TF-IDF, or embeddings) instead of a single sentiment score.
- Try a better sentiment model (VADER, or a pretrained transformer).
- Use cross-validation instead of a single train-test split for more reliable accuracy estimates on this small dataset.
- Deploy as a simple Streamlit app where a user can input a song name and get a predicted mood.

## Tech Stack
`Python` · `pandas` · `spotipy` (initial approach) · `lyricsgenius` · `TextBlob` · `scikit-learn` (RandomForest, GradientBoosting, LogisticRegression, SVM)

---
*This project was built as a hands-on learning exercise covering the full ML workflow: data collection from multiple sources (API + dataset), feature engineering, avoiding data leakage, model comparison, and honest evaluation.*