import pandas as pd
import numpy as np
import glob
import re
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from scipy.signal import medfilt  # <-- DODANY IMPORT



# --- STAŁE KONFIGURACYJNE ---
DATA_FOLDER = "TESTNoc"
MODEL_FILENAME = "activity_classifier_model2.joblib"
SEGMENT_LENGTH = 250
SAMPLE_RATE = 50

# --- FUNKCJE POMOCNICZE ---

def extract_number(filename):
    """Wyodrębnia numer z nazwy pliku IMU."""
    basename = os.path.basename(filename)
    match = re.search(r'IMU(\d+)\.csv$', basename)
    return int(match.group(1)) if match else float('inf')

def apply_median_filter(df, window_size=3):
    """
    Stosuje filtr medianowy do każdej kolumny DataFrame w celu usunięcia
    pojedynczych pików (szumu typu "sól i pieprz").

    Args:
        df (pd.DataFrame): Ramka danych do filtrowania.
        window_size (int): Rozmiar okna filtra. Musi być nieparzysty.
                           Małe wartości (3, 5) są zalecane do usuwania
                           pojedynczych błędów bez zbytniego wygładzania sygnału.

    Returns:
        pd.DataFrame: Ramka danych po filtracji.
    """
    print(f"Stosowanie filtru medianowego z oknem o rozmiarze {window_size}...")
    
    # Filtr medianowy z scipy.signal wymaga nieparzystego rozmiaru okna
    if window_size % 2 == 0:
        window_size += 1
        print(f"  Dostosowano rozmiar okna do nieparzystej wartości: {window_size}")

    df_filtered = df.copy()
    for col in df.columns:
        # Upewniamy się, że dane są typu float przed filtrowaniem
        df_filtered[col] = medfilt(df[col].astype(float), kernel_size=window_size)
    
    print("Filtracja medianowa zakończona.")
    return df_filtered

def napraw_bledy_IMU(df):
    """
    Przeszukuje DataFrame w poszukiwaniu błędów w danych AKCELEROMETRU.
    Jeśli w danym wierszu co najmniej dwie wartości z akcelerometru są
    identyczne, cały wiersz (wszystkie 9 osi) jest zastępowany
    średnią z sąsiednich, poprawnych wierszy.
    """
    if df.shape[1] < 9:
        print("Ostrzeżenie: Dane mają mniej niż 9 kolumn. Funkcja naprawy może nie działać poprawnie.")
        return df
        
    df_fixed = df.copy().astype(float)
    print("Rozpoczynam wyszukiwanie i naprawę błędów na podstawie akcelerometru...")
    errors_found = 0
    
    acc_start, acc_end = 3, 6
    
    for i in range(1, len(df_fixed) - 1):
        acc_values = df_fixed.iloc[i, acc_start:acc_end]
        
        if pd.notna(acc_values).all() and acc_values.nunique() < 3:
            errors_found += 1
            prev_row = df_fixed.iloc[i - 1]
            next_row = df_fixed.iloc[i + 1]
            mean_row = (prev_row + next_row) / 2
            df_fixed.iloc[i] = mean_row

    if errors_found > 0:
        print(f"Znaleziono i naprawiono {errors_found} wierszy na podstawie błędów w akcelerometrze.")
    else:
        print("Nie znaleziono błędów w danych akcelerometru.")
        
    return df_fixed

def extract_features(segment):
    """Wyodrębnia cechy z segmentu danych."""
    features = []
    for i in range(segment.shape[1]):
        axis_data = segment[:, i]
        features.append(np.mean(axis_data))
        features.append(np.std(axis_data))
        features.append(np.var(axis_data))
        features.append(np.min(axis_data))
        features.append(np.max(axis_data))
        features.append(np.median(axis_data))
        features.append(skew(axis_data))
        features.append(kurtosis(axis_data))
        features.append(np.quantile(axis_data, 0.75) - np.quantile(axis_data, 0.25))
        features.append(np.sum(np.square(axis_data)))
        N = len(axis_data)
        yf = rfft(axis_data)
        xf = rfftfreq(N, 1 / SAMPLE_RATE)
        if len(yf) > 1:
            dominant_freq_idx = np.argmax(np.abs(yf[1:])) + 1
            dominant_freq = xf[dominant_freq_idx]
            dominant_freq_magnitude = np.abs(yf[dominant_freq_idx])
        else:
            dominant_freq = 0
            dominant_freq_magnitude = 0
        features.append(dominant_freq)
        features.append(dominant_freq_magnitude)
    return features

def load_start_time_from_file(data_folder, default_time="12:00:00"):
    """Wczytuje czas rozpoczęcia z pliku time.txt."""
    time_file_path = os.path.join(data_folder, "time.txt")
    try:
        with open(time_file_path, 'r') as f:
            time_str = f.read().strip()
            start_timestamp = pd.to_datetime(time_str, format='%d/%m/%Y %H:%M:%S')
            print(f"Pomyślnie wczytano czas rozpoczęcia z pliku: {start_timestamp}")
            return start_timestamp
    except (FileNotFoundError, ValueError):
        print(f"OSTRZEŻENIE: Nie znaleziono lub niepoprawny format pliku 'time.txt'. Używam domyślnej godziny.")
        return pd.to_datetime(f"1970-01-01 {default_time}")

# --- GŁÓWNA CZĘŚĆ SKRYPTU ---

# 1. PRZYGOTOWANIE DANYCH
print("Krok 1: Wczytywanie i przygotowanie danych...")
file_list = sorted(glob.glob(os.path.join(DATA_FOLDER, "IMU*.csv")), key=extract_number)

# Wczytujemy dane, od razu zamieniając myślniki '-' na braki danych (NaN)
all_data_frames = [pd.read_csv(file, na_values=['-']) for file in file_list]
combined_df = pd.concat(all_data_frames, ignore_index=True)

# Krok 1a: Uzupełnienie brakujących danych (NaN) za pomocą interpolacji liniowej
print("Uzupełnianie brakujących danych metodą interpolacji...")
combined_df.interpolate(method='linear', limit_direction='both', inplace=True)

# Krok 1b: ZASTOSOWANIE FILTRU MEDIANOWEGO
# Usuwa pojedyncze, gwałtowne skoki wartości (outliery).
combined_df = apply_median_filter(combined_df, window_size=3)

# Krok 1c: Naprawa błędów specyficznych dla IMU (Twoja oryginalna funkcja)
combined_df = napraw_bledy_IMU(combined_df)

print(f"Wczytano i przygotowano łącznie {len(combined_df)} próbek z {len(file_list)} plików.")

# 1.5. Tworzenie osi czasu
print("\nKrok 1.5: Tworzenie osi czasu...")
start_time = load_start_time_from_file(DATA_FOLDER) 
time_index = pd.to_timedelta(np.arange(len(combined_df)) / SAMPLE_RATE, unit='s')
combined_df.index = start_time + time_index
print(f"Oś czasu utworzona. Czas trwania pomiaru: {combined_df.index[-1] - combined_df.index[0]}")


# 2. KLASYFIKACJA "KROK PO KROKU"
print("\nKrok 2: Klasyfikacja segmentów...")
try:
    model = joblib.load(MODEL_FILENAME)
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku modelu '{MODEL_FILENAME}'. Zakończono działanie.")
    exit()

step = SEGMENT_LENGTH
predictions = []
data_np = combined_df.to_numpy() # Użyj to_numpy() dla lepszej wydajności

for i in range(0, len(data_np) - SEGMENT_LENGTH + 1, step):
    segment = data_np[i : i + SEGMENT_LENGTH]
    features = extract_features(segment)
    predicted_label = model.predict([features])[0]
    predictions.append({
        "start_idx": i,
        "end_idx": i + SEGMENT_LENGTH,
        "label": predicted_label
    })

print(f"Sklasyfikowano {len(predictions)} segmentów.")

# 2.5. PODSUMOWANIE CZASU AKTYWNOŚCI
print("\n--- Podsumowanie Czasu Aktywności (w konsoli) ---")
activity_counts = {}
for pred in predictions:
    activity_counts[pred['label']] = activity_counts.get(pred['label'], 0) + 1

segment_duration_seconds = SEGMENT_LENGTH / SAMPLE_RATE
total_classified_seconds = 0

for label, count in sorted(activity_counts.items()):
    total_time_seconds = count * segment_duration_seconds
    total_classified_seconds += total_time_seconds
    minutes = int(total_time_seconds // 60)
    seconds = int(total_time_seconds % 60)
    print(f"- {label.capitalize():<10}: {count} segmentów, co daje {minutes} min {seconds} s")

total_minutes = int(total_classified_seconds // 60)
total_seconds = int(total_classified_seconds % 60)
print("------------------------------------------")
print(f"Łączny sklasyfikowany czas: {total_minutes} min {total_seconds} s")

# 3. WIZUALIZACJA Z PODSUMOWANIEM
print("\nKrok 3: Generowanie wykresu z podsumowaniem...")

label_to_color = {
    'siedzenie': 'orange',
    'leżenie': 'gold',
    'ruch': 'mediumpurple',
}

fig, ax = plt.subplots(figsize=(20, 10))

# Rysowanie sygnałów akcelerometru
ax.plot(combined_df.index, combined_df.iloc[:, 3], label='Przyspieszenie X (AX)', color='red', alpha=0.7)
ax.plot(combined_df.index, combined_df.iloc[:, 4], label='Przyspieszenie Y (AY)', color='blue', alpha=0.7)
ax.plot(combined_df.index, combined_df.iloc[:, 5], label='Przyspieszenie Z (AZ)', color='green', alpha=0.9)

# Rysowanie obszarów aktywności
for pred in predictions:
    start_idx, end_idx, label = pred['start_idx'], pred['end_idx'], pred['label']
    color = label_to_color.get(label, 'black')
    
    # Upewnij się, że indeksy nie wychodzą poza zakres
    if start_idx < len(combined_df.index):
        time_start = combined_df.index[start_idx]
        time_end = combined_df.index[min(end_idx, len(combined_df.index) - 1)]
        ax.axvspan(time_start, time_end, color=color, alpha=0.3, lw=0)

# Przygotowanie tekstu do podsumowania na wykresie
summary_lines = []
for label, count in sorted(activity_counts.items()):
    total_time_seconds = count * segment_duration_seconds
    minutes = int(total_time_seconds // 60)
    seconds = int(total_time_seconds % 60)
    summary_lines.append(f"- {label.capitalize():<10}: {minutes} min {seconds} s")

summary_text = "Podsumowanie czasu aktywności:\n" + \
               "\n".join(summary_lines) + \
               "\n" + "-"*28 + "\n" + \
               f"Łącznie: {total_minutes} min {total_seconds} s"

props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

# Legenda
activity_legend_patches = [mpatches.Patch(color=color, label=label, alpha=0.4) for label, color in label_to_color.items()]
signal_legend = ax.legend(loc='upper left', title="Sygnały")
ax.add_artist(signal_legend)
ax.legend(handles=activity_legend_patches, loc='upper right', title="Sklasyfikowane aktywności")

# Ustawienia wykresu i formatowanie osi czasu
ax.set_title('Wykres aktywności z automatyczną klasyfikacją', fontsize=16)
ax.set_xlabel('Czas (HH:MM:SS)', fontsize=12)
ax.set_ylabel('Przyspieszenie', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)

formatter = mdates.DateFormatter('%H:%M:%S')
ax.xaxis.set_major_formatter(formatter)
fig.autofmt_xdate(rotation=30) # Lekka rotacja dla lepszej czytelności

# plt.tight_layout()
plt.show()