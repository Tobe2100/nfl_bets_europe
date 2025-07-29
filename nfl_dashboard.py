@st.cache_data # Cache-Funktion, um Ladezeiten zu optimieren
def load_data(file_path):
    """Lädt die NFL-Spieldaten und passt Spaltennamen an."""
    if not os.path.exists(file_path):
        st.error(f"Fehler: Die Datei '{file_path}' wurde nicht gefunden. Bitte stelle sicher, dass sie im selben Verzeichnis liegt.")
        st.stop() # Stoppt die Ausführung der App

    df = pd.read_csv(file_path)

    # Spalten umbenennen, um sie an deinen Modellcode anzupassen
    # WICHTIG: Die Linke Seite ist der Name in deiner CSV, die Rechte Seite ist der Name, den dein Code erwartet
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
    # Da deine "Date"-Spalte nur 'MM/DD' enthält, müssen wir ein Jahr hinzufügen.
    # Basierend auf deiner Beispielzeile '2017' nehmen wir das Jahr der Saison.
    # ABER: Die Spalte 'Season' in deiner CSV ist viel besser! Lass uns das nutzen.
    try:
        # Kombiniere 'Season' und 'Date' für ein vollständiges Datum
        # Beispiel: '2017' + '/' + '09/03' -> '2017/09/03'
        df['game_date'] = pd.to_datetime(df['Season'].astype(str) + '/' + df['game_date'], format='%Y/%m/%d', errors='coerce')
    except Exception as e:
        st.error(f"Fehler bei der Datumsumwandlung: {e}. Überprüfe das Format der 'Date'-Spalte oder 'Season'-Spalte.")
        st.stop()


    # Entferne Zeilen, bei denen die Datumsumwandlung fehlgeschlagen ist
    df.dropna(subset=['game_date'], inplace=True)

    return df