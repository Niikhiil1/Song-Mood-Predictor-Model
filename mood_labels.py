import pandas as pd
from textblob import TextBlob

df = pd.read_csv('spotify_audio_dataset_with_lyrics.csv')

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

df['lyrics_sentiment'] = df['lyrics'].apply(get_sentiment)


def get_mood(valence , sentiment):
    if valence > 0.5 and sentiment >= 0:
        return 'happy'
    elif valence <= 0.5 and sentiment <= 0:
        return 'sad'
    else:
        return 'neutral'


df['mood'] = df.apply(lambda row: get_mood(row['valence'], row['lyrics_sentiment']), axis = 1)


df.to_csv('final_spotify_audio_dataset_with_lyrics_and_mood.csv', index = False)

print(df['mood'].value_counts())