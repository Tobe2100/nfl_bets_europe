import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

st.set_page_config(layout="wide")
st.title("🏈 NFL Prognose-Dashboard")
st.write("Willkommen zu deinem personalisierten NFL Spielprognose-Tool!")

# --- Hilfsfunktionen für Daten und Modell ---

@st.cache_data # Cache-Funktion, um Ladezeiten zu optimieren
def load_data(file_path):
    """Lädt die NFL-Spieldaten und passt Spaltennamen an."""
    if not os.path.exists(file_path):
        st.error(f"Fehler: Die Datei '{file_path}' wurde nicht gefunden. Bitte stelle sicher, dass sie im selben Verzeichnis liegt.")
        st.stop() # Stoppt die Ausführung der App

    # Liest die CSV-Datei. Pandas sollte das Komma als Trennzeichen automatisch erkennen.
    df = pd.read_csv(file_path)

    # Spalten umbenennen, um sie an deinen Modellcode anzupassen
    df = df.rename(columns={
        'Date': 'game_date',       # 'Date' aus deiner Datei wird zu 'game_date'
        'HomeTeam': 'home_team',   # 'HomeTeam' aus deiner Datei wird zu 'home_team'
        'AwayTeam': 'away_team',   # 'AwayTeam' aus deiner Datei wird zu 'away_team'
        'HomeScore': 'home_score', # 'HomeScore' aus deiner Datei wird zu 'home_score'
        'AwayScore': 'away_score'  # 'AwayScore' aus deiner Datei wird zu 'away_score'
    })

    # Nur die benötigten Spalten auswählen
    required_columns = ['game_date', 'home_team', 'away_team', 'home_score', 'away_score']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        st.error(f"Fehler: Die Datei '{file_path}' enthält nicht alle benötigten Spalten nach dem Umbenennen. Fehlend: {missing_cols}")
        st.stop()

    df = df[required_columns] # Nur die relevanten Spalten behalten

    # Sicherstellen, dass die Score-Spalten numerisch sind und fehlende Werte (NaN) handhaben
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')

    # Zeilen mit fehlenden Scores entfernen
    df.dropna(subset=['home_score', 'away_score'], inplace=True)
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)

    # --- WICHTIGE ÄNDERUNG HIER: Datumsumwandlung ---
    # Füge das Jahr aus der 'Season'-Spalte hinzu, um ein vollständiges Datum zu erstellen
    # (ANNAHME: 'Date' ist im Format 'MM/DD' und 'Season' ist das Jahr)
    try:
        # Kombiniere 'Season' und 'Date' und wandle um
        # Hier nehmen wir an, dass die "Season"-Spalte existiert und das Jahr enthält.
        # WENN NICHT, müssen wir die Strategie anpassen (z.B. ein Standardjahr wie 2017 annehmen).
        # BASIERT AUF deiner zuvor geteilten Dateistruktur: Season,Week,GameStatus,Day,Date...
        df['game_date'] = pd.to_datetime(
            df['Season'].astype(str) + '/' + df['game_date'],
            format='%Y/%m/%d',
            errors='coerce'
        )
    except KeyError:
        st.error("Fehler: 'Season' Spalte fehlt für die Datumsumwandlung. Oder 'Date' ist nicht im Format MM/DD.")
        st.stop()
    except Exception as e:
        st.error(f"Fehler bei der Datumsumwandlung: {e}. Überprüfe das Format der 'Date' und 'Season'-Spalten.")
        st.stop()

    # Entferne Zeilen, bei denen die Datumsumwandlung fehlgeschlagen ist
    df.dropna(subset=['game_date'], inplace=True)

    return df

@st.cache_data
def compute_power_rankings(df_games):
    """Berechnet vereinfachte Power Rankings basierend auf erzielten/zugelassenen Punkten."""
    teams = pd.unique(df_games[['home_team', 'away_team']].values.ravel())
    power = pd.DataFrame(index=teams, columns=['offense_pts_avg', 'defense_pts_avg'])

    for team in teams:
        home_scored = df_games[df_games['home_team'] == team]['home_score']
        away_scored = df_games[df_games['away_team'] == team]['away_score']
        points_scored = pd.concat([home_scored, away_scored])

        home_allowed = df_games[df_games['home_team'] == team]['away_score']
        away_allowed = df_games[df_games['away_team'] == team]['home_score']
        points_allowed = pd.concat([home_allowed, away_allowed])

        power.loc[team, 'offense_pts_avg'] = points_scored.mean() if not points_scored.empty else 0
        power.loc[team, 'defense_pts_avg'] = points_allowed.mean() if not points_allowed.empty else 0

    power = power.fillna(0)
    return power

@st.cache_data
def create_features_and_labels(df_games, power_rankings_df):
    """Erstellt Features und Labels für das Modell."""
    features = []
    labels = []

    for idx, row in df_games.iterrows():
        home = row['home_team']
        away = row['away_team']

        home_off = power_rankings_df.loc[home, 'offense_pts_avg'] if home in power_rankings_df.index else 0
        home_def = power_rankings_df.loc[home, 'defense_pts_avg'] if home in power_rankings_df.index else 0
        away_off = power_rankings_df.loc[away, 'offense_pts_avg'] if away in power_rankings_df.index else 0
        away_def = power_rankings_df.loc[away, 'defense_pts_avg'] if away in power_rankings_df.index else 0

        home_power = home_off - away_def
        away_power = away_off - home_def
        home_advantage = 1

        features.append([home_power, away_power, home_advantage])
        labels.append(1 if row['home_score'] > row['away_score'] else 0)

    return np.array(features), np.array(labels)

@st.cache_resource
def train_model(X_train, y_train):
    """Trainiert das Random Forest Klassifikator Modell."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# --- Hauptlogik der Streamlit App ---

# 1. Daten laden
df_games = load_data('nfl_games.csv')
st.sidebar.header("Einstellungen")

if df_games is not None:
    st.success("NFL-Spieldaten erfolgreich geladen!")

    # 2. Power Rankings berechnen
    power_rankings = compute_power_rankings(df_games)

    # 3. Features und Labels erstellen
    X, y = create_features_and_labels(df_games, power_rankings)

    # 4. Modell trainieren
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = train_model(X_train, y_train)

    # 5. Modell bewerten (optional in der App anzeigen)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Modell-Performance")
    st.write(f"Die Genauigkeit des Modells auf den Testdaten beträgt: **{accuracy:.2f}**")

    st.markdown("---")

    # --- Power Rankings anzeigen ---
    st.subheader("Aktuelle Power Rankings (vereinfacht)")
    sort_column = st.sidebar.selectbox(
        "Sortiere Power Rankings nach:",
        ['offense_pts_avg', 'defense_pts_avg'],
        index=0
    )
    st.dataframe(power_rankings.sort_values(by=sort_column, ascending=False))

    st.markdown("---")

    # --- Prognosen für zukünftige Spiele (Beispiel) ---
    st.subheader("Prognose für ein Spiel")

    all_teams = sorted(power_rankings.index.tolist())
    home_team_selection = st.selectbox("Heimteam auswählen:", all_teams)
    away_team_selection = st.selectbox("Auswärtsteam auswählen:", [t for t in all_teams if t != home_team_selection])

    if st.button("Prognose erstellen"):
        if home_team_selection and away_team_selection:
            home_off = power_rankings.loc[home_team_selection, 'offense_pts_avg']
            home_def = power_rankings.loc[home_team_selection, 'defense_pts_avg']
            away_off = power_rankings.loc[away_team_selection, 'offense_pts_avg']
            away_def = power_rankings.loc[away_team_selection, 'defense_pts_avg']

            game_features = np.array([[
                home_off - away_def,
                away_off - home_def,
                1 # Heimvorteil
            ]])

            prediction_proba = model.predict_proba(game_features)[0]

            home_win_proba = prediction_proba[1]
            away_win_proba = prediction_proba[0]

            st.write(f"**Prognose für das Spiel {home_team_selection} vs. {away_team_selection}:**")
            st.write(f"Wahrscheinlichkeit Heimsieg ({home_team_selection}): **{home_win_proba:.2f}**")
            st.write(f"Wahrscheinlichkeit Auswärtssieg ({away_team_selection}): **{away_win_proba:.2f}**")

            if home_win_proba > away_win_proba:
                st.success(f"Das Modell prognostiziert einen Sieg für die **{home_team_selection}**!")
            else:
                st.info(f"Das Modell prognostiziert einen Sieg für die **{away_team_selection}**!")
        else:
            st.warning("Bitte wähle beide Teams aus, um eine Prognose zu erhalten.")
