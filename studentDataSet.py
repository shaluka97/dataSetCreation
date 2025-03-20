import lyricsgenius
import csv
import time
import requests
from tqdm import tqdm  # For progress tracking
import re  # For regular expressions to clean lyrics

# --- Configuration (Keep original values) ---
GENIUS_API_TOKEN = "ZOBbiN15EypwcgYaRYcogdslS7Il9Ou34tJI0ARIZflNvJGJmYa3JxpvV6lAKhO8"
TARGET_GENRE = "House"
OUTPUT_CSV = "Student_dataset.csv"
TARGET_SONGS = 100
DELAY_BETWEEN_REQUESTS = 1

# --- Enhanced Artist List ---
artists = [
     "Frankie Knuckles"
    # ... Add more artists ...
]

# --- Initialize LyricsGenius with better error handling ---
try:
    genius = lyricsgenius.Genius(GENIUS_API_TOKEN,
                                 skip_non_songs=True,
                                 excluded_terms=["(Remix)", "(Live)", "(Cover)", "Instrumental"],
                                 remove_section_headers=True)
    genius.verbose = False  # Reduce console output clutter
except Exception as e:
    print(f"Failed to initialize Genius API client: {e}")
    exit(1)


def clean_lyrics(lyrics):
    """Clean lyrics text to remove problematic characters and formatting"""
    if not lyrics:
        return ""

    # Remove Genius artifacts
    lyrics = re.sub(r'EmbedShare URLCopyEmbedCopy', '', lyrics)

    # Remove section headers if any remain
    lyrics = re.sub(r'\[.*?\]', '', lyrics)

    # Replace newlines with a space
    lyrics = lyrics.replace('\n', ' ')

    # Replace tabs with spaces (since we're using tab as delimiter)
    lyrics = lyrics.replace('\t', ' ')

    # Replace multiple spaces with a single space
    lyrics = re.sub(r'\s+', ' ', lyrics)

    # Strip whitespace
    lyrics = lyrics.strip()

    return lyrics


def get_song_data(artist_name):
    """Gets song data with improved error handling and genre filtering"""
    song_data = []
    retries = 3

    while retries > 0:
        try:
            artist = genius.search_artist(artist_name, max_songs=10, sort="popularity")
            if artist is None:
                print(f"Artist '{artist_name}' not found.")
                return []

            for song in artist.songs:
                try:
                    print(f"Processing song: {song.title}")

                    # Add delay to respect rate limits
                    time.sleep(DELAY_BETWEEN_REQUESTS)

                    # Search for the specific song
                    genius_song = genius.search_song(song.title, artist.name)

                    # Skip if song retrieval failed
                    if not genius_song:
                        print(f"Could not find detailed information for {song.title}")
                        continue

                    # Extract and validate release year
                    release_year = None
                    if hasattr(genius_song, 'year') and genius_song.year:
                        try:
                            release_year = int(genius_song.year.split('-')[0])

                        except (ValueError, AttributeError):
                            pass

                    # Validate lyrics
                    lyrics = genius_song.lyrics
                    if not lyrics or len(lyrics) < 10:  # Skip songs with no/minimal lyrics
                        print(f"Skipping {song.title} - insufficient lyrics")
                        continue

                    # Clean up lyrics
                    lyrics = clean_lyrics(lyrics)

                    # Store the validated data
                    song_data.append({
                        "artist_name": artist.name,
                        "track_name": genius_song.title,
                        "release_date": release_year,
                        "genre": TARGET_GENRE,
                        "lyrics": lyrics
                    })
                    print(f"Added song: {genius_song.title}")

                except requests.exceptions.RequestException as e:
                    print(f"Network error when retrieving {song.title}: {e}")
                    time.sleep(2)  # Wait longer on network errors
                except AttributeError as e:
                    print(f"Attribute error for song {song.title}: {e}")
                    # Continue with next song
                except Exception as e:
                    print(f"Error processing song {song.title}: {e}")

            # Successfully got data, break retry loop
            break

        except requests.exceptions.RequestException as e:
            print(f"Network error when retrieving artist {artist_name}: {e}")
            retries -= 1
            if retries > 0:
                print(f"Retrying {artist_name} in 5 seconds... ({retries} attempts left)")
                time.sleep(5)
        except Exception as e:
            print(f"Error getting data for {artist_name}: {e}")
            retries -= 1
            if retries > 0:
                print(f"Retrying {artist_name} in 5 seconds... ({retries} attempts left)")
                time.sleep(5)

    return song_data


def save_to_csv(data, filename):
    """Saves the collected data to a CSV file with proper formatting"""
    if not data:
        print("No data to save!")
        return False

    try:
        # Use comma delimiter instead of tab to avoid issues with tab characters in the lyrics
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["artist_name", "track_name", "release_date", "genre", "lyrics"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()

            for row in data:
                # Ensure all values are strings and properly escaped
                cleaned_row = {k: str(v) if v is not None else "" for k, v in row.items()}
                writer.writerow(cleaned_row)

        return True
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False


def main():
    all_song_data = []
    collected_songs = set()  # Track unique songs to avoid duplicates

    print(f"Starting collection of {TARGET_SONGS} songs...")

    # Use tqdm for a progress bar
    for artist in tqdm(artists, desc="Processing artists"):
        print(f"\nFetching songs for {artist}...")
        song_data = get_song_data(artist)

        # Add only non-duplicate songs based on artist+title
        for song in song_data:
            song_key = f"{song['artist_name']}|{song['track_name']}"
            if song_key not in collected_songs:
                all_song_data.append(song)
                collected_songs.add(song_key)
                print(f"Added: {song['track_name']} by {song['artist_name']}")
            else:
                print(f"Skipped duplicate: {song['track_name']}")

        print(f"Current song count: {len(all_song_data)}")
        if len(all_song_data) >= TARGET_SONGS:
            print(f"Reached target of {TARGET_SONGS} songs.")
            break

    # Summary before saving
    print("\n--- Collection Summary ---")
    print(f"Total songs collected: {len(all_song_data)}")

    artist_counts = {}
    for song in all_song_data:
        artist = song["artist_name"]
        artist_counts[artist] = artist_counts.get(artist, 0) + 1

    print("\nSongs per artist:")
    for artist, count in artist_counts.items():
        print(f"- {artist}: {count} songs")

    # Save to CSV
    if save_to_csv(all_song_data, OUTPUT_CSV):
        print(f"\nSuccessfully saved data to {OUTPUT_CSV}")
    else:
        print(f"\nFailed to save data to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
