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

    # Liest die CSV-Datei.
    df = pd.read_csv(file_path)

    # Spalten umbenennen, um sie an deinen Modellcode anzupassen
    # Die Linke Seite ist der Name, der EXAKT in deiner CSV-Datei steht
    # Die Rechte Seite ist der Name, den der Rest deines Python-Codes erwartet
    df = df.rename(columns={
        'Date': 'game_date',        # "Date" aus deiner CSV wird zu "game_date"
        'HomeTeam': 'home_team',    # "HomeTeam" aus deiner CSV wird zu "home_team"
        'AwayTeam': 'away_team',    # "AwayTeam" aus deiner CSV wird zu "away_team"
        'HomeScore': 'home_score',  # "HomeScore" aus deiner CSV wird zu "home_score"
        'AwayScore': 'away_score'   # "AwayScore" aus deiner CSV wird zu "away_score"
    })

    # Nur die benötigten Spalten auswählen (jetzt mit den umbenannten Namen)
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
    # Kombiniere 'Season' und 'Date' für ein vollständiges Datum, da 'Date' nur MM/DD ist
    # und deine CSV eine 'Season'-Spalte enthält.
    try:
        # Füge das Jahr aus der 'Season'-Spalte zum Monat/Tag hinzu
        df['game_date'] = pd.to_datetime(df['Season'].astype(str) + '/' + df['game_date'], format='%Y/%m/%d', errors='coerce')
    except KeyError:
        # Falls die 'Season'-Spalte fehlen sollte (unwahrscheinlich bei deiner CSV)
        st.error("Fehler: Die 'Season'-Spalte fehlt in der Datenquelle, die für die Datumsumwandlung benötigt wird.")
        st.stop()
    except Exception as e:
        st.error(f"Fehler bei der Datumsumwandlung: {e}. Überprüfe das Format der 'Date'-Spalte oder 'Season'-Spalte.")
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

        # Überprüfen, ob die Teams in den Power Rankings sind, sonst 0 verwenden
        home_off = power_rankings_df.loc[home, 'offense_pts_avg'] if home in power_rankings_df.index else 0
        home_def = power_rankings_df.loc[home, 'defense_pts_avg'] if home in power_rankings_df.index else 0
        away_off = power_rankings_df.loc[away, 'offense_pts_avg'] if away in power_rankings_df.index else 0
        away_def = power_rankings_df.loc[away, 'defense_pts_avg'] if away in power_rankings_df.index else 0

        home_power = home_off - away_def
        away_power = away_off - home_def
        home_advantage = 1 # Annahme: Heimteam hat immer Heimvorteil

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
# Stelle sicher, dass 'nfl_games.csv' im selben Verzeichnis wie diese App liegt
df_games = load_data('nfl_games.csv')
st.sidebar.header("Einstellungen")

if df_games is not None and not df_games.empty:
    st.success("NFL-Spieldaten erfolgreich geladen!")

    # 2. Power Rankings berechnen
    power_rankings = compute_power_rankings(df_games)

    # Überprüfen, ob Power Rankings berechnet werden konnten
    if power_rankings.empty or power_rankings.isnull().all().all():
        st.error("Fehler: Konnte keine Power Rankings aus den Daten berechnen. Sind genug Spiele vorhanden und die Teamnamen korrekt?")
        st.stop()

    # 3. Features und Labels erstellen
    X, y = create_features_and_labels(df_games, power_rankings)

    if len(X) == 0:
        st.error("Fehler: Keine Features oder Labels für das Training des Modells erstellt. Überprüfe die Daten und die Power Rankings.")
        st.stop()

    # 4. Modell trainieren
    # Sicherstellen, dass genügend Daten für Trainings- und Testsets vorhanden sind
    if len(X) < 2: # Mindestens 2 Samples für train_test_split (eher mehr)
        st.error("Nicht genügend Daten für das Modelltraining. Bitte lade mehr Spieldaten hoch.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = train_model(X_train, y_train)

    # 5. Modell bewerten (optional in der App anzeigen)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Modell-Performance")
    st.write(f"Die Genauigkeit des Modells auf den Testdaten beträgt: **{accuracy:.2f}**")

    st.markdown("---")

    # --- Power Rankings anzeigen ---
    st.subheader("Aktuelle Power Rankings (vereinfacht)")
    # Sicherstellen, dass Power Rankings nicht leer sind, bevor sie angezeigt werden
    if not power_rankings.empty:
        sort_column = st.sidebar.selectbox(
            "Sortiere Power Rankings nach:",
            ['offense_pts_avg', 'defense_pts_avg'],
            index=0
        )
        st.dataframe(power_rankings.sort_values(by=sort_column, ascending=False))
    else:
        st.info("Power Rankings konnten noch nicht berechnet werden.")


    st.markdown("---")

    # --- Prognosen für zukünftige Spiele (Beispiel) ---
    st.subheader("Prognose für ein Spiel")

    all_teams = sorted(power_rankings.index.tolist())
    # Filtern der Teams, die auch tatsächlich Power Rankings haben (keine 0-Werte, wenn möglich)
    all_teams = [team for team in all_teams if power_rankings.loc[team, 'offense_pts_avg'] > 0 or power_rankings.loc[team, 'defense_pts_avg'] > 0]
    all_teams = sorted(list(set(all_teams))) # Einzigartige und sortierte Liste

    if not all_teams:
        st.warning("Keine Teams verfügbar, um eine Prognose zu erstellen. Bitte stelle sicher, dass deine Daten Teams und Spielergebnisse enthalten.")
    else:
        home_team_selection = st.selectbox("Heimteam auswählen:", all_teams)
        away_team_selection = st.selectbox("Auswärtsteam auswählen:", [t for t in all_teams if t != home_team_selection])

        if st.button("Prognose erstellen"):
            if home_team_selection and away_team_selection:
                # Sicherstellen, dass die ausgewählten Teams in den Power Rankings sind
                if home_team_selection not in power_rankings.index or away_team_selection not in power_rankings.index:
                    st.error("Eines der ausgewählten Teams ist nicht in den Power Rankings. Das Modell kann keine Prognose erstellen.")
                else:
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

                    home_win_proba = prediction_proba[1] # Wahrscheinlichkeit für Klasse 1 (Heimsieg)
                    away_win_proba = prediction_proba[0] # Wahrscheinlichkeit für Klasse 0 (Auswärtssieg)

                    st.write(f"**Prognose für das Spiel {home_team_selection} vs. {away_team_selection}:**")
                    st.write(f"Wahrscheinlichkeit Heimsieg ({home_team_selection}): **{home_win_proba:.2f}**")
                    st.write(f"Wahrscheinlichkeit Auswärtssieg ({away_team_selection}): **{away_win_proba:.2f}**")

                    if home_win_proba > away_win_proba:
                        st.success(f"Das Modell prognostiziert einen Sieg für die **{home_team_selection}**!")
                    else:
                        st.info(f"Das Modell prognostiziert einen Sieg für die **{away_team_selection}**!")
            else:
                st.warning("Bitte wähle beide Teams aus, um eine Prognose zu erhalten.")
else:
    st.error("Fehler: Konnte keine Spieldaten laden oder die Datei ist leer. Bitte überprüfe 'nfl_games.csv'.")
    st.info("Stelle sicher, dass 'nfl_games.csv' die Spalten 'Season', 'Date', 'HomeTeam', 'AwayTeam', 'HomeScore', 'AwayScore' enthält.")