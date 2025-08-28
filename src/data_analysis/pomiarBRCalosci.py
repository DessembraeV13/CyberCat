import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, find_peaks, correlate
import os
import joblib
import re
import glob
from scipy.ndimage import binary_dilation, label
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
import matplotlib.dates as mdates

# --- KONFIGURACJA GŁÓWNA ---
IMU_FOLDER = "TESTNoc"
BR_FOLDER = "TESTNoc"
MODEL_FILENAME = "activity_classifier_model.joblib"
SAMPLE_RATE = 50
SEGMENT_LENGTH_IMU = 250
MOTION_LABEL = 'ruch'
SAFETY_MARGIN_SECONDS = 2.0
F_LOW, F_HIGH, NUM_TAPS = 0.1, 1.0, 101
FILES_PER_BLOCK = 4
RECORDING_INTERVAL_MINUTES = 5
# Parametry detekcji pików
PEAK_MIN_DISTANCE = 25 
PEAK_MIN_HEIGHT = 0.3  
PEAK_MIN_PROMINENCE = 0.2

# --- FUNKCJE POMOCNICZE (bez zmian) ---
def natural_sort_key(s):
    match = re.search(r'(\d+)', os.path.basename(s))
    return int(match.group(1)) if match else float('inf')

def napraw_bledy_IMU(df):
    if df.shape[1] < 9: return df
    df_fixed = df.copy().astype(float)
    errors_found = 0; acc_start, acc_end = 3, 6
    for i in range(1, len(df_fixed) - 1):
        acc_values = df_fixed.iloc[i, acc_start:acc_end]
        if pd.notna(acc_values).all() and acc_values.nunique() < 3:
            errors_found += 1
            prev_row, next_row = df_fixed.iloc[i - 1], df_fixed.iloc[i + 1]
            mean_row = (prev_row + next_row) / 2; df_fixed.iloc[i] = mean_row
    if errors_found > 0: print(f"  - Naprawiono {errors_found} wierszy IMU.")
    return df_fixed

def extract_features(segment):
    features = []
    for i in range(segment.shape[1]):
        axis_data = segment[:, i]
        features.extend([np.mean(axis_data), np.std(axis_data), np.var(axis_data), np.min(axis_data), np.max(axis_data), np.median(axis_data), skew(axis_data), kurtosis(axis_data), np.quantile(axis_data, 0.75) - np.quantile(axis_data, 0.25), np.sum(np.square(axis_data))])
        N, yf, xf = len(axis_data), rfft(axis_data), rfftfreq(len(axis_data), 1 / SAMPLE_RATE)
        if len(yf) > 1:
            dominant_freq_idx = np.argmax(np.abs(yf[1:])) + 1
            features.extend([xf[dominant_freq_idx], np.abs(yf[dominant_freq_idx])])
        else: features.extend([0, 0])
    return features

def load_start_time(folder, default_time="12:00:00"):
    path = os.path.join(folder, "time.txt")
    try:
        with open(path, 'r') as f: return pd.to_datetime(f.read().strip(), format='%d/%m/%Y %H:%M:%S')
    except (FileNotFoundError, ValueError): return pd.to_datetime(f"1970-01-01 {default_time}")

# --- GŁÓWNA LOGIKA SKRYPTU ---

print("Krok 1: Wyszukiwanie i grupowanie bloków pomiarowych...")
br_files = sorted(glob.glob(os.path.join(BR_FOLDER, "BR*.csv")), key=natural_sort_key)
if not br_files: print("BŁĄD: Nie znaleziono plików BR*.csv."); exit()
model = joblib.load(MODEL_FILENAME)
start_timestamp = load_start_time(BR_FOLDER)
try:
    motion_class_index = list(model.classes_).index(MOTION_LABEL)
except ValueError:
    print(f"BŁĄD: Etykieta '{MOTION_LABEL}' nie została znaleziona w klasach modelu. Zakończono."); exit()

bpm_results = []
last_block_data_for_plot = {}

for i in range(0, len(br_files), FILES_PER_BLOCK):
    block_br_files = br_files[i : i + FILES_PER_BLOCK]
    if len(block_br_files) < FILES_PER_BLOCK:
        print(f"\nOstrzeżenie: Niekompletny blok na końcu ({len(block_br_files)}/{FILES_PER_BLOCK} plików). Pomijanie.")
        continue
        
    block_index = i // FILES_PER_BLOCK
    print(f"\n--- Przetwarzanie bloku pomiarowego nr {block_index} ---")
    
    # Wczytywanie i mapowanie
    block_imu_dfs, block_br_dfs, valid_block = [], [], True
    for j, br_path in enumerate(block_br_files):
        imu_file_num = (block_index * 10) + j
        imu_path = os.path.join(IMU_FOLDER, f"IMU{imu_file_num}.csv")
        if os.path.exists(imu_path):
            print(f"  - Parowanie: {os.path.basename(br_path)} -> {os.path.basename(imu_path)}")
            df_br, df_imu = pd.read_csv(br_path, na_values=['-']), pd.read_csv(imu_path, na_values=['-'])
            df_br.interpolate(method='linear', limit_direction='both', inplace=True)
            df_imu.interpolate(method='linear', limit_direction='both', inplace=True)
            min_len = min(len(df_br), len(df_imu))
            block_br_dfs.append(df_br.head(min_len)); block_imu_dfs.append(napraw_bledy_IMU(df_imu.head(min_len)))
        else:
            print(f"BŁĄD: Brak pliku IMU{imu_file_num}.csv dla bloku. Ten blok zostanie pominięty."); valid_block = False; break
            
    if not valid_block or not block_br_dfs: continue
    
    block_br_df = pd.concat(block_br_dfs, ignore_index=True)
    block_imu_df = pd.concat(block_imu_dfs, ignore_index=True)
    
    # === POCZĄTEK ZAAWANSOWANEGO POTOKU PRZETWARZANIA ===
    
    # Klasyfikacja ruchu (tylko AI)
    initial_motion_mask = np.ones(len(block_imu_df), dtype=bool)
    motion_probabilities = np.zeros(len(block_imu_df))
    for k in range(0, len(block_imu_df) - SEGMENT_LENGTH_IMU + 1, SEGMENT_LENGTH_IMU):
        segment = block_imu_df.iloc[k : k + SEGMENT_LENGTH_IMU].values
        features = extract_features(segment)
        label = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        if label == MOTION_LABEL: initial_motion_mask[k : k + SEGMENT_LENGTH_IMU] = False
        motion_probabilities[k : k + SEGMENT_LENGTH_IMU] = probabilities[motion_class_index]

    # Dynamiczna synchronizacja i finalna maska
    imu_motion_signal = motion_probabilities
    br_motion_proxy = np.abs(np.diff(block_br_df["BR"].values, prepend=0))
    if np.ptp(imu_motion_signal) > 0:
        imu_motion_signal = (imu_motion_signal - np.min(imu_motion_signal)) / np.ptp(imu_motion_signal)
    if np.ptp(br_motion_proxy) > 0:
        br_motion_proxy = (br_motion_proxy - np.min(br_motion_proxy)) / np.ptp(br_motion_proxy)
    correlation = correlate(br_motion_proxy, imu_motion_signal, mode='same', method='fft')
    delay_samples = np.argmax(correlation) - (len(imu_motion_signal) // 2)
    shifted_mask = np.roll(initial_motion_mask, delay_samples)
    safety_margin_samples = int(SAFETY_MARGIN_SECONDS * SAMPLE_RATE)
    dilated_motion_areas = binary_dilation(~shifted_mask, structure=np.ones(safety_margin_samples))
    final_motion_mask = ~dilated_motion_areas

    # Analiza oddechu
    br_signal = block_br_df["BR"].values
    bandpass_coeffs = firwin(NUM_TAPS, [F_LOW, F_HIGH], fs=SAMPLE_RATE, pass_zero='bandpass')
    filtered_br_initial = filtfilt(bandpass_coeffs, 1.0, br_signal)
    baseline = pd.Series(filtered_br_initial).rolling(window=3*SAMPLE_RATE, center=True, min_periods=1).mean().values
    detrended_br = filtered_br_initial - baseline
    local_amp = pd.Series(detrended_br).rolling(window=3*SAMPLE_RATE, center=True, min_periods=1).std().values
    normalized_br = detrended_br / (local_amp + 1e-6)
    rectified_br = np.where(normalized_br > 0, normalized_br, 0)
    energy_br = np.square(rectified_br)
    masked_energy_br = energy_br.copy(); masked_energy_br[~final_motion_mask] = np.nan
    peaks, _ = find_peaks(masked_energy_br, distance=PEAK_MIN_DISTANCE, height=PEAK_MIN_HEIGHT, prominence=PEAK_MIN_PROMINENCE)
    
    # Obliczenie i zapisanie BPM dla bloku
    clean_duration_min = (np.sum(final_motion_mask) / SAMPLE_RATE) / 60
    if clean_duration_min > 0.1: # Analizuj, jeśli jest chociaż trochę (>6s) czystych danych
        chunk_bpm = len(peaks) / clean_duration_min
        chunk_center_time = start_timestamp + pd.to_timedelta((block_index * RECORDING_INTERVAL_MINUTES * 60) + (len(block_br_df)/SAMPLE_RATE/2), unit='s')
        bpm_results.append({'time': chunk_center_time, 'bpm': chunk_bpm})
        print(f"  - Wynik dla bloku: {chunk_bpm:.2f} oddechów/min")
        
        last_block_data_for_plot = {
            'br_signal': detrended_br,
            'imu_signal': block_imu_df,
            'energy_signal': masked_energy_br,
            'peaks': peaks,
            'final_motion_mask': final_motion_mask
        }

# === KROK 2: Podsumowanie i Wizualizacja ===
print("\nKrok 2: Podsumowanie i wizualizacja...")
if bpm_results:
    bpm_df = pd.DataFrame(bpm_results)
    overall_bpm = bpm_df['bpm'].mean()
    
    print(f"\n--- OGÓLNE PODSUMOWANIE ANALIZY ODDECHU ---")
    print(f"Pomyślnie przeanalizowano {len(bpm_results)} z {len(br_files)//FILES_PER_BLOCK} bloków pomiarowych.")
    print(f"Średnia częstość oddechów (w okresach bezruchu): {overall_bpm:.2f} oddechów/minutę")

    fig, ax_bpm = plt.subplots(figsize=(20, 7))
    ax_bpm.set_title("Częstość oddechów w czasie", fontsize=16)
    ax_bpm.axhspan(15, 30, color='green', alpha=0.15, label='Norma oddechowa (15-30)')
    ax_bpm.axhline(15, color='green', linestyle='--', linewidth=1); ax_bpm.axhline(30, color='green', linestyle='--', linewidth=1)
    ax_bpm.axhline(15, color='blue', linestyle=':', linewidth=1.5, label='Bradypnoe (<15)')
    ax_bpm.axhline(30, color='red', linestyle=':', linewidth=1.5, label='Tachypnoe (>30)')
    ax_bpm.plot(bpm_df['time'], bpm_df['bpm'], marker='o', linestyle='-', color='black', label="Częstość oddechu (BR)")
    ax_bpm.set_ylabel("Oddechy na minutę"); ax_bpm.set_xlabel("Czas")
    ax_bpm.grid(True, which='both', linestyle=':')
    handles, labels = ax_bpm.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax_bpm.legend(unique_labels.values(), unique_labels.keys(),loc='lower right')
    
    formatter = mdates.DateFormatter('%H:%M')
    ax_bpm.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    
    # plt.tight_layout(); 
    plt.show()
else:
    print("Nie udało się obliczyć częstości oddechu dla żadnego bloku.")

if last_block_data_for_plot:
    print("\nGenerowanie wykresu diagnostycznego dla ostatniego przetworzonego bloku...")
    data = last_block_data_for_plot
    fig, (ax1, ax_energy) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    fig.suptitle(f"Szczegółowa Analiza Ostatniego Bloku", fontsize=16)
    time_axis = np.arange(len(data['br_signal'])) / SAMPLE_RATE
    
    ax1.set_title("Sygnał Oddechowy (BR)")
    ax1.plot(time_axis, data['br_signal'], label="Sygnał BR (przetworzony)", color='darkred')
    ax1.plot(time_axis[data['peaks']], data['br_signal'][data['peaks']], "x", color='blue', markersize=8, label=f"Wykryte oddechy")
    ax1.legend(loc='upper right'); ax1.grid(True); ax1.set_ylabel("Wartość")
    
    ax_energy.set_title("Znormalizowany Sygnał Energii")
    ax_energy.plot(time_axis, data['energy_signal'], label="Energia sygnału", color='purple')
    ax_energy.plot(time_axis[data['peaks']], data['energy_signal'][data['peaks']], "x", color='cyan', markersize=8)
    ax_energy.legend(loc='upper right'); ax_energy.grid(True); ax_energy.set_ylabel("Energia"); ax_energy.set_xlabel("Czas (s)")

    motion_indices = np.where(~data['final_motion_mask'])[0]
    if len(motion_indices) > 0:
        labeled_array, num_features = label(~data['final_motion_mask'])
        for i in range(1, num_features + 1):
            indices = np.where(labeled_array == i)[0]
            start_time, end_time = indices[0] / SAMPLE_RATE, indices[-1] / SAMPLE_RATE
            ax1.axvspan(start_time, end_time, color='red', alpha=0.15, lw=0)
            ax_energy.axvspan(start_time, end_time, color='red', alpha=0.15, lw=0)

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95]); 
    plt.show()