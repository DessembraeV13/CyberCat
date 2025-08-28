import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, find_peaks, correlate
import os
import joblib
import re
from scipy.ndimage import binary_dilation, label
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq

# --- KONFIGURACJA GŁÓWNA ---
IMU_FOLDER = "brzuch2"
BR_FOLDER = "brzuch2"
MODEL_FILENAME = "activity_classifier_model.joblib"
START_BR_INDEX = 0
NUM_BR_FILES = 4
SAMPLE_RATE = 50
SEGMENT_LENGTH_IMU = 250
MOTION_LABEL = 'ruch'
SAFETY_MARGIN_SECONDS = 2.0
F_LOW = 0.1
F_HIGH = 1.0
NUM_TAPS = 101
# === NOWA KONFIGURACJA: Parametry detekcji pików ===
PEAK_MIN_DISTANCE = 25 
PEAK_MIN_HEIGHT = 0.3  
PEAK_MIN_PROMINENCE = 0.2 # Minimalna "wydatność" piku

# --- FUNKCJE POMOCNICZE (bez zmian) ---
def map_br_to_imu_index(br_index):
    block_index = br_index // NUM_BR_FILES
    imu_start_index = block_index * 10
    offset_in_block = br_index % NUM_BR_FILES
    return imu_start_index + offset_in_block

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

# --- GŁÓWNA LOGIKA SKRYPTU ---

# === KROK 1 & 2 (bez zmian) ===
print("Krok 1: Wczytywanie i łączenie danych...")
all_imu_dfs, all_br_dfs = [], []
for i in range(START_BR_INDEX, START_BR_INDEX + NUM_BR_FILES):
    imu_index = map_br_to_imu_index(i)
    br_path, imu_path = os.path.join(BR_FOLDER, f"BR{i}.csv"), os.path.join(IMU_FOLDER, f"IMU{imu_index}.csv")
    if os.path.exists(br_path) and os.path.exists(imu_path):
        print(f"  - Parowanie: {os.path.basename(br_path)} -> {os.path.basename(imu_path)}")
        df_br, df_imu = pd.read_csv(br_path, na_values=['-']), pd.read_csv(imu_path, na_values=['-'])
        df_br.interpolate(method='linear', limit_direction='both', inplace=True)
        df_imu.interpolate(method='linear', limit_direction='both', inplace=True)
        min_len = min(len(df_br), len(df_imu))
        all_br_dfs.append(df_br.head(min_len)); all_imu_dfs.append(napraw_bledy_IMU(df_imu.head(min_len)))
if not all_br_dfs: print("BŁĄD: Nie wczytano plików."); exit()
combined_br_df, combined_imu_df = pd.concat(all_br_dfs, ignore_index=True), pd.concat(all_imu_dfs, ignore_index=True)

print("\nKrok 2: Klasyfikacja ruchu za pomocą modelu AI...")
model = joblib.load(MODEL_FILENAME)
try:
    motion_class_index = list(model.classes_).index(MOTION_LABEL)
except ValueError:
    print(f"BŁĄD: Etykieta '{MOTION_LABEL}' nie została znaleziona w klasach modelu. Zakończono."); exit()
initial_motion_mask, motion_probabilities = np.ones(len(combined_imu_df), dtype=bool), np.zeros(len(combined_imu_df))
for i in range(0, len(combined_imu_df) - SEGMENT_LENGTH_IMU + 1, SEGMENT_LENGTH_IMU):
    segment, features = combined_imu_df.iloc[i : i + SEGMENT_LENGTH_IMU].values, []
    features = extract_features(segment)
    label, probabilities = model.predict([features])[0], model.predict_proba([features])[0]
    if label == MOTION_LABEL: initial_motion_mask[i : i + SEGMENT_LENGTH_IMU] = False
    motion_probabilities[i : i + SEGMENT_LENGTH_IMU] = probabilities[motion_class_index]

# === KROK 3: Dynamiczna synchronizacja i finalna maska (bez zmian) ===
print("\nKrok 3: Dynamiczna synchronizacja i tworzenie finalnej maski...")
imu_motion_signal, br_motion_proxy = motion_probabilities, np.abs(np.diff(combined_br_df["BR"].values, prepend=0))
if np.ptp(imu_motion_signal) > 0:
    imu_motion_signal = (imu_motion_signal - np.min(imu_motion_signal)) / np.ptp(imu_motion_signal)
if np.ptp(br_motion_proxy) > 0:
    br_motion_proxy = (br_motion_proxy - np.min(br_motion_proxy)) / np.ptp(br_motion_proxy)
correlation = correlate(br_motion_proxy, imu_motion_signal, mode='same', method='fft')
delay_samples = np.argmax(correlation) - (len(imu_motion_signal) // 2)
delay_seconds = delay_samples / SAMPLE_RATE
if delay_seconds > 0: print(f"  Synchronizacja: Sygnał BR jest opóźniony o {delay_seconds:.3f} s.")
else: print(f"  Synchronizacja: Sygnał IMU jest opóźniony o {-delay_seconds:.3f} s.")
shifted_mask = np.roll(initial_motion_mask, delay_samples)
safety_margin_samples = int(SAFETY_MARGIN_SECONDS * SAMPLE_RATE)
dilated_motion_areas = binary_dilation(~shifted_mask, structure=np.ones(safety_margin_samples))
final_motion_mask = ~dilated_motion_areas

# === KROK 4: Analiza Oddechu ===
print("\nKrok 4: Analiza oddechu...")
br_signal = combined_br_df["BR"].values
bandpass_coeffs = firwin(NUM_TAPS, [F_LOW, F_HIGH], fs=SAMPLE_RATE, pass_zero='bandpass')
filtered_br_initial = filtfilt(bandpass_coeffs, 1.0, br_signal)
baseline_window_size = 3 * SAMPLE_RATE
baseline = pd.Series(filtered_br_initial).rolling(window=baseline_window_size, center=True, min_periods=1).mean().values
detrended_br = filtered_br_initial - baseline
local_amplitude = pd.Series(detrended_br).rolling(window=baseline_window_size, center=True, min_periods=1).std().values
epsilon = 1e-6
normalized_br = detrended_br / (local_amplitude + epsilon)
rectified_br = np.where(normalized_br > 0, normalized_br, 0)
energy_br = np.square(rectified_br)
masked_energy_br = energy_br.copy()
masked_energy_br[~final_motion_mask] = np.nan

# === TUTAJ ZMIANA: Dodanie parametru 'prominence' ===
peaks, _ = find_peaks(masked_energy_br, 
                      distance=PEAK_MIN_DISTANCE, 
                      height=PEAK_MIN_HEIGHT, 
                      prominence=PEAK_MIN_PROMINENCE)
# === KONIEC ZMIANY ===

clean_signal_samples = np.sum(final_motion_mask)
clean_signal_duration_minutes = (clean_signal_samples / SAMPLE_RATE) / 60
breath_count = len(peaks)
breaths_per_minute = breath_count / clean_signal_duration_minutes if clean_signal_duration_minutes > 0 else 0
print(f"Wykryta liczba oddechów: {breath_count}, Częstotliwość oddechu: {breaths_per_minute:.2f} oddechów/min")

# === KROK 5: Wizualizacja (bez zmian) ===
print("\nKrok 5: Generowanie wykresu...")
num_samples = len(detrended_br)
time_axis = np.arange(num_samples) / SAMPLE_RATE
fig, (ax1, ax_energy) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
fig.suptitle(f"Analiza Oddechu (BR ≈ {breaths_per_minute:.1f} oddechów/min)", fontsize=16)
ax1.set_title("Sygnał Oddechowy (BR)")
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.plot(time_axis, detrended_br, label="Sygnał BR (przetworzony)", color='darkred')
ax1.plot(time_axis[peaks], detrended_br[peaks], "x", color='blue', markersize=8, label=f"Wykryte oddechy ({breath_count})")
ax1.set_ylabel("Wartość sygnału")
ax1.legend(loc='upper right')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax_energy.set_title("Sygnał Energii (użyty do detekcji pików)")
ax_energy.plot(time_axis, masked_energy_br, label="Energia sygnału", color='purple')
ax_energy.plot(time_axis[peaks], masked_energy_br[peaks], "x", color='cyan', markersize=8)
ax_energy.set_ylabel("Znormalizowana energia")
ax_energy.set_xlabel("Czas (s)")
ax_energy.legend(loc='upper right')
ax_energy.grid(True, which='both', linestyle='--', linewidth=0.5)
motion_indices = np.where(~final_motion_mask)[0]
if len(motion_indices) > 0:
    labeled_array, num_features = label(~final_motion_mask)
    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        start_time, end_time = indices[0] / SAMPLE_RATE, indices[-1] / SAMPLE_RATE
        ax1.axvspan(start_time, end_time, color='red', alpha=0.15, lw=0)
        ax_energy.axvspan(start_time, end_time, color='red', alpha=0.15, lw=0)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()