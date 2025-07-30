import pandas as pd
import glob
import os

# --- Configuration ---
# Assuming your CSV files are in the same directory as this script.
# If they are in a subfolder (e.g., 'data/'), change this path accordingly.
CSV_DIRECTORY = '.'

# --- Data Loading ---

def load_nfl_games_data(directory):
    """
    Loads the nfl_games.csv file and performs initial preprocessing.
    Calculates win percentages for away and home teams.
    """
    file_path = os.path.join(directory, 'nfl_games.csv')
    try:
        df_games = pd.read_csv(file_path)
        print(f"Successfully loaded: {os.path.basename(file_path)}")

        # --- Initial Feature Engineering for nfl_games ---
        # Convert AwayRecord and HomeRecord to win percentages
        # Example: "4-0" -> 4 wins, 0 losses. Win % = 4 / (4+0) = 1.0
        # We need to handle cases where record might be "0-0" or invalid.

        def calculate_win_percentage(record_str):
            if pd.isna(record_str) or record_str == '0-0':
                return 0.0 # Or NaN, depending on how you want to handle no games played yet
            try:
                wins, losses = map(int, record_str.split('-'))
                total_games = wins + losses
                if total_games == 0:
                    return 0.0 # Avoid division by zero if record is like "0-0"
                return wins / total_games
            except ValueError:
                return 0.0 # Handle unexpected record formats gracefully
            except AttributeError:
                return 0.0 # Handle non-string inputs if any

        df_games['AwayWinPct'] = df_games['AwayRecord'].apply(calculate_win_percentage)
        df_games['HomeWinPct'] = df_games['HomeRecord'].apply(calculate_win_percentage)

        # Convert AwayScore and HomeScore to numeric, coercing errors to NaN
        df_games['AwayScore'] = pd.to_numeric(df_games['AwayScore'], errors='coerce')
        df_games['HomeScore'] = pd.to_numeric(df_games['HomeScore'], errors='coerce')

        # Fill NaN scores with 0 for games that might not have scores yet (e.g., future games)
        df_games['AwayScore'] = df_games['AwayScore'].fillna(0)
        df_games['HomeScore'] = df_games['HomeScore'].fillna(0)


        print("Added 'AwayWinPct' and 'HomeWinPct' features.")
        print("Cleaned 'AwayScore' and 'HomeScore'.")
        return df_games

    except FileNotFoundError:
        print(f"Error: '{os.path.basename(file_path)}' not found. Make sure the file is in the correct directory.")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        print(f"An error occurred loading {os.path.basename(file_path)}: {e}")
        return pd.DataFrame()

def load_nfl_plays_data(directory):
    """
    Loads all *_plays.csv files and concatenates them into a single DataFrame.
    """
    all_plays_dfs = []
    plays_files = glob.glob(os.path.join(directory, '*_plays.csv'))

    if not plays_files:
        print(f"No *_plays.csv files found in '{directory}'.")
        return pd.DataFrame()

    print(f"Found {len(plays_files)} plays CSV files. Loading them...")
    for file_path in plays_files:
        try:
            df = pd.read_csv(file_path)
            all_plays_dfs.append(df)
            # Optional: Add a source column to know which file a row came from
            # df['SourceFile'] = os.path.basename(file_path)
            print(f"  - Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  - Error loading {os.path.basename(file_path)}: {e}")

    if all_plays_dfs:
        combined_df = pd.concat(all_plays_dfs, ignore_index=True)
        print("\nAll plays CSVs combined successfully!")
        return combined_df
    else:
        print("\nNo plays CSV files were successfully loaded.")
        return pd.DataFrame()

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Loading NFL Games Data ---")
    nfl_games_df = load_nfl_games_data(CSV_DIRECTORY)

    if not nfl_games_df.empty:
        print("\n--- NFL Games Data Head ---")
        print(nfl_games_df.head())
        print("\n--- NFL Games Data Info ---")
        nfl_games_df.info()

    print("\n\n--- Loading NFL Plays Data ---")
    nfl_plays_df = load_nfl_plays_data(CSV_DIRECTORY)

    if not nfl_plays_df.empty:
        print("\n--- NFL Plays Data Head ---")
        print(nfl_plays_df.head())
        print("\n--- NFL Plays Data Info ---")
        nfl_plays_df.info()

    print("\n\n--- Next Steps ---")
    print("1. You now have 'nfl_games_df' (with win percentages) and 'nfl_plays_df' loaded.")
    print("2. The next crucial step is to **acquire actual betting spread data** for NFL games.")
    print("   Without this, you cannot compare your model's forecasts to find 'value'.")
    print("3. Once you have spread data, we can proceed with more advanced feature engineering")
    print("   (e.g., aggregating play-by-play data to game level, creating rolling averages),")
    print("   model training, and the value comparison logic.")
