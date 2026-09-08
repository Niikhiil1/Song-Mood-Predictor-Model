

import pandas as pd 
import lyricsgenius as lg
import time

Genius_acess_token = "your_genius_access_token_here"  # Replace with your actual Genius API access token

genius = lg.Genius(Genius_acess_token, timeout = 15, retries = 2)
genius.verbose = False
genius.remove_section_headers = True



def get_lyrics(artist_name, song_name):
    try:
        song = genius.search_song(song_name, artist_name)
        if song:
            return song.lyrics
    except Exception as e:
        print(f"Could not find the song {song_name} by {artist_name}    : {e}")



def build_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    df = df[['track_name', 'artists', 'danceability', 'energy', 'valence', 'tempo', 'acousticness', 'instrumentalness', 'loudness', 'speechiness', 'track_genre']]

    df.drop_duplicates(subset = ['track_name', 'artists'], inplace = True)


    df = df.sample(n = 250, random_state = 42)
    
    empty_list = []
    for i, row in df.iterrows():
        artist_name = row['artists']
        song_name = row['track_name']
        lyrics = get_lyrics(artist_name, song_name)
        empty_list.append(lyrics)
    df['lyrics'] = empty_list

    df = df.dropna(subset = ['lyrics'])
    df.to_csv(output_file, index = False)



build_dataset('spotify_audio_features.csv', 'spotify_audio_dataset_with_lyrics.csv')