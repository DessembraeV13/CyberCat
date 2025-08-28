import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, medfilt, correlate
import heartpy as hp
import os
import joblib
import re
from scipy.ndimage import binary_dilation
import glob
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
import matplotlib.dates as mdates

# --- KONFIGURACJA GŁÓWNA ---
IMU_FOLDER = "TESTNoc"
PPG_FOLDER = "TESTNoc"
MODEL_FILENAME = "activity_classifier_model.joblib"
SAMPLE_RATE_PPG, SAMPLE_RATE_IMU = 200, 50
SEGMENT_LENGTH_IMU = 250
MOTION_LABEL = 'ruch'
LPF_CUTOFF, HPF_CUTOFF, NUM_TAPS = 6, 1, 201
VARIANCE_WINDOW_SIZE, VARIANCE_THRESHOLD = int(0.5 * SAMPLE_RATE_IMU), 5000
SAFETY_MARGIN_SECONDS_PPG = 3
REJECT_RATIO_THRESHOLD = 0.4
INITIAL_TRIM_SECONDS = 0.5
FILES_PER_BLOCK = 4
RECORDING_INTERVAL_MINUTES = 5
# Konfiguracja filtra wygładzającego
SMOOTHING_LPF_CUTOFF = 10.0
SMOOTHING_NUM_TAPS = 51

# --- FUNKCJE POMOCNICZE ---
def natural_sort_key(s):
    match = re.search(r'(\d+)', os.path.basename(s))
    return int(match.group(1)) if match else float('inf')

def apply_median_filter(df, window_size=3):
    if window_size % 2 == 0: window_size += 1
    df_filtered = df.copy()
    for col in df.columns:
        df_filtered[col] = medfilt(df[col].astype(float), kernel_size=window_size)
    return df_filtered

def napraw_bledy_IMU(df):
    if df.shape[1] < 9: return df
    df_fixed = df.copy().astype(float)
    errors_found = 0; acc_start, acc_end = 3, 6
    for i in range(1, len(df_fixed) - 1):
        acc_values = df_fixed.iloc[i, acc_start:acc_end]
        if pd.notna(acc_values).all() and acc_values.nunique() < 3:
            errors_found += 1; prev_row, next_row = df_fixed.iloc[i - 1], df_fixed.iloc[i + 1]
            mean_row = (prev_row + next_row) / 2; df_fixed.iloc[i] = mean_row
    if errors_found > 0: print(f"    - Naprawiono {errors_found} wierszy IMU.")
    return df_fixed

def extract_features(segment):
    features = []
    for i in range(segment.shape[1]):
        axis_data = segment[:, i]; features.extend([np.mean(axis_data), np.std(axis_data), np.var(axis_data), np.min(axis_data), np.max(axis_data), np.median(axis_data), skew(axis_data), kurtosis(axis_data), np.quantile(axis_data, 0.75) - np.quantile(axis_data, 0.25), np.sum(np.square(axis_data))])
        N, yf, xf = len(axis_data), rfft(axis_data), rfftfreq(len(axis_data), 1 / SAMPLE_RATE_IMU)
        if len(yf) > 1: dominant_freq_idx = np.argmax(np.abs(yf[1:])) + 1; features.extend([xf[dominant_freq_idx], np.abs(yf[dominant_freq_idx])])
        else: features.extend([0, 0])
    return features

def load_start_time(folder, default_time="12:00:00"):
    path = os.path.join(folder, "time.txt")
    try:
        with open(path, 'r') as f: return pd.to_datetime(f.read().strip(), format='%d/%m/%Y %H:%M:%S')
    except (FileNotFoundError, ValueError): return pd.to_datetime(f"1970-01-01 {default_time}")

# --- GŁÓWNA LOGIKA SKRYPTU ---

print("Krok 1: Wyszukiwanie i grupowanie bloków pomiarowych...")
ppg_files = sorted(glob.glob(os.path.join(PPG_FOLDER, "PPG*.csv")), key=natural_sort_key)
if not ppg_files: print("BŁĄD: Nie znaleziono plików PPG*.csv."); exit()
try:
    model = joblib.load(MODEL_FILENAME)
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku modelu '{MODEL_FILENAME}'."); exit()

start_timestamp = load_start_time(PPG_FOLDER)
bpm_results, last_block_data_for_plot = [], {}
safety_margin_samples_ppg = int(SAFETY_MARGIN_SECONDS_PPG * SAMPLE_RATE_PPG)

for i in range(0, len(ppg_files), FILES_PER_BLOCK):
    block_ppg_files = ppg_files[i : i + FILES_PER_BLOCK]
    if len(block_ppg_files) < FILES_PER_BLOCK: print(f"\nOstrzeżenie: Niekompletny blok. Pomijanie."); continue

    try:
        first_ppg_num = int(re.search(r'(\d+)', os.path.basename(block_ppg_files[0])).group(1))
    except (AttributeError, ValueError):
        print(f"Ostrzeżenie: Nie można odczytać numeru z pliku {os.path.basename(block_ppg_files[0])}. Pomijanie bloku.")
        continue
    
    print(f"\n--- Przetwarzanie bloku zaczynającego się od PPG{first_ppg_num}.csv ---")
    
    block_index = i // FILES_PER_BLOCK
    imu_block_start_index = block_index * 10
    print(f"  Logika mapowania: Blok PPG {first_ppg_num}-{first_ppg_num+3} -> Blok IMU {imu_block_start_index}-{imu_block_start_index+3}")
    
    block_imu_dfs, block_ppg_dfs, valid_block = [], [], True
    for j, ppg_path in enumerate(block_ppg_files):
        imu_file_num = imu_block_start_index + j
        imu_path = os.path.join(IMU_FOLDER, f"IMU{imu_file_num}.csv")
        
        if os.path.exists(ppg_path) and os.path.exists(imu_path):
            print(f"    - Parowanie: {os.path.basename(ppg_path)} -> {os.path.basename(imu_path)}")
            df_ppg = pd.read_csv(ppg_path, na_values=['-']).interpolate(method='linear', limit_direction='both')
            df_imu = pd.read_csv(imu_path, na_values=['-']).interpolate(method='linear', limit_direction='both')
            df_imu = apply_median_filter(df_imu)
            min_len_sec = min(len(df_ppg)/SAMPLE_RATE_PPG, len(df_imu)/SAMPLE_RATE_IMU)
            block_ppg_dfs.append(df_ppg.head(int(min_len_sec*SAMPLE_RATE_PPG)))
            block_imu_dfs.append(napraw_bledy_IMU(df_imu.head(int(min_len_sec*SAMPLE_RATE_IMU))))
        else: 
            print(f"BŁĄD: Brak pasującego pliku {os.path.basename(imu_path)}."); valid_block = False; break
    if not valid_block: continue
    
    block_ppg_df, block_imu_df = pd.concat(block_ppg_dfs, ignore_index=True), pd.concat(block_imu_dfs, ignore_index=True)
    trim_samples_ppg, trim_samples_imu = int(INITIAL_TRIM_SECONDS * SAMPLE_RATE_PPG), int(INITIAL_TRIM_SECONDS * SAMPLE_RATE_IMU)
    if len(block_ppg_df) > trim_samples_ppg and len(block_imu_df) > trim_samples_imu:
        block_ppg_df, block_imu_df = block_ppg_df.iloc[trim_samples_ppg:].reset_index(drop=True), block_imu_df.iloc[trim_samples_imu:].reset_index(drop=True)
        print(f"  Przycięto początkowe {INITIAL_TRIM_SECONDS}s sygnału.")
    
    # === Pełny, zaawansowany potok przetwarzania sygnału ===
    motion_mask_ai = np.ones(len(block_imu_df), dtype=bool)
    for k in range(0, len(block_imu_df) - SEGMENT_LENGTH_IMU + 1, SEGMENT_LENGTH_IMU):
        if model.predict([extract_features(block_imu_df.iloc[k:k+SEGMENT_LENGTH_IMU].values)])[0] == MOTION_LABEL:
            motion_mask_ai[k : k + SEGMENT_LENGTH_IMU] = False
    rolling_variance = block_imu_df.iloc[:, 3:6].rolling(window=VARIANCE_WINDOW_SIZE).var().sum(axis=1).fillna(0)
    final_motion_mask_imu = np.logical_and(motion_mask_ai, rolling_variance < VARIANCE_THRESHOLD)

    ir_signal = block_ppg_df["IR"].values
    filtered_ir = filtfilt(firwin(NUM_TAPS, LPF_CUTOFF, fs=SAMPLE_RATE_PPG), 1.0, ir_signal)
    filtered_ir = filtfilt(firwin(NUM_TAPS, HPF_CUTOFF, fs=SAMPLE_RATE_PPG, pass_zero="highpass"), 1.0, filtered_ir)
    filtered_ir = hp.flip_signal(filtered_ir)
    detrended_ir = filtered_ir - np.mean(filtered_ir)
    signal_before_correction = detrended_ir.copy()

    imu_motion_signal = rolling_variance.to_numpy()
    ppg_motion_signal = np.abs(np.diff(filtered_ir, prepend=0))
    if np.ptp(imu_motion_signal) > 0:
        imu_motion_signal = (imu_motion_signal - np.min(imu_motion_signal)) / np.ptp(imu_motion_signal)
    if np.ptp(ppg_motion_signal) > 0:
        ppg_motion_signal = (ppg_motion_signal - np.min(ppg_motion_signal)) / np.ptp(ppg_motion_signal)
    resampling_factor = int(SAMPLE_RATE_PPG / SAMPLE_RATE_IMU)
    imu_motion_resampled = np.repeat(imu_motion_signal, resampling_factor)
    min_len = min(len(ppg_motion_signal), len(imu_motion_resampled))
    ppg_motion_signal, imu_motion_resampled = ppg_motion_signal[:min_len], imu_motion_resampled[:min_len]
    correlation = correlate(ppg_motion_signal, imu_motion_resampled, mode='same', method='fft')
    delay_samples_ppg = np.argmax(correlation) - (len(imu_motion_resampled) // 2)
    delay_samples_imu = int(delay_samples_ppg / resampling_factor)
    shifted_mask_imu = np.roll(final_motion_mask_imu, delay_samples_imu)

    motion_mask_ppg_narrow = np.repeat(shifted_mask_imu, resampling_factor)
    target_len, current_len = len(detrended_ir), len(motion_mask_ppg_narrow)
    if current_len > target_len: motion_mask_ppg_narrow = motion_mask_ppg_narrow[:target_len]
    elif current_len < target_len: motion_mask_ppg_narrow = np.pad(motion_mask_ppg_narrow, (0, target_len - current_len), mode='edge')
    safety_margin_samples = int(SAFETY_MARGIN_SECONDS_PPG * SAMPLE_RATE_PPG)
    dilated_motion_areas = binary_dilation(~motion_mask_ppg_narrow, structure=np.ones(safety_margin_samples))
    final_motion_mask_ppg = ~dilated_motion_areas

    corrected_ir = detrended_ir.copy()
    motion_indices = np.where(~final_motion_mask_ppg)[0]
    if len(motion_indices) > 0:
        motion_start_idx, motion_end_idx = motion_indices[0], motion_indices[-1]
        stable_window_samples = int(1.0 * SAMPLE_RATE_PPG)
        if motion_start_idx > stable_window_samples and motion_end_idx < len(detrended_ir) - stable_window_samples:
            mean_before = np.mean(detrended_ir[motion_start_idx - stable_window_samples : motion_start_idx])
            mean_after = np.mean(detrended_ir[motion_end_idx + 1 : motion_end_idx + 1 + stable_window_samples])
            dc_offset = mean_after - mean_before
            corrected_ir[motion_end_idx + 1:] -= dc_offset
    
    corrected_ir[~final_motion_mask_ppg] = np.nan
    interpolated_ir = pd.Series(corrected_ir).interpolate(method='linear', limit_direction='both').bfill().ffill().values
    interpolated_ir = interpolated_ir - np.mean(interpolated_ir)
    final_signal = filtfilt(firwin(SMOOTHING_NUM_TAPS, SMOOTHING_LPF_CUTOFF, fs=SAMPLE_RATE_PPG), 1.0, interpolated_ir)
    
    # === Analiza HeartPy ===
    try:
        working_data, measures = hp.process(final_signal, SAMPLE_RATE_PPG, interp_threshold=1000)
        
        if 'bpm' in measures and pd.notna(measures['bpm']):
            num_rejected, num_accepted = len(working_data.get('removed_beats',[])), len(working_data.get('peaklist',[]))
            if (num_rejected + num_accepted) > 0 and (num_rejected / (num_rejected + num_accepted) <= REJECT_RATIO_THRESHOLD):
                block_start_offset_min = (first_ppg_num // FILES_PER_BLOCK) * RECORDING_INTERVAL_MINUTES
                block_center_time = start_timestamp + pd.to_timedelta(block_start_offset_min, unit='m')
                
                bpm_results.append({'time': block_center_time, 'bpm': measures['bpm'], 'rmssd': measures['rmssd']})
                print(f"  - Pomiar ZAAKCEPTOWANY. BPM: {measures['bpm']:.2f}, RMSSD: {measures['rmssd']:.2f}")
                last_block_data_for_plot = {
                    'ppg_before_correction': signal_before_correction,
                    'ppg_final': final_signal,
                    'imu': block_imu_df,
                    'mask': final_motion_mask_ppg,
                    'variance': rolling_variance
                }
            else: 
                print("  - Pomiar ODRZUCONY (niska jakość sygnału HeartPy - za dużo odrzuconych pików).")
        else: 
            print("  - Pomiar ODRZUCONY (HeartPy nie obliczyło BPM - wynik to 'nan').")
    except Exception as e: 
        print(f"  - BŁĄD HeartPy: {e}. Pomiar ODRZUCONY.")

# === Podsumowanie i Wizualizacja ===
print("\nKrok 2: Podsumowanie i wizualizacja...")
if bpm_results:
    bpm_df = pd.DataFrame(bpm_results)
    overall_bpm = bpm_df['bpm'].mean()
    overall_rmssd = bpm_df['rmssd'].mean()
    
    print(f"\n--- OGÓLNE PODSUMOWANIE ANALIZY TĘTNA ---")
    print(f"Pomyślnie przeanalizowano {len(bpm_results)} z {len(ppg_files)//FILES_PER_BLOCK} bloków pomiarowych.")
    print(f"Średnie Tętno (w okresach bezruchu): {overall_bpm:.2f} BPM")
    print(f"Średnia Zmienność Rytmu Serca (w okresach bezruchu): {overall_rmssd:.2f} ms (RMSSD)")

    # === TUTAJ PIERWSZA POPRAWKA ===
    # Tworzymy figurę i osie w jednej komendzie, przechwytując zmienną 'fig'
    fig, ax_bpm = plt.subplots(figsize=(20, 7))
    
    ax_bpm.set_title("Wartość tętna w czasie", fontsize=16)
    ax_bpm.axhspan(120, 200, color='green', alpha=0.15, label='Norma spoczynkowa (120-200)')
    ax_bpm.axhline(120, color='green', linestyle='--', linewidth=1); ax_bpm.axhline(200, color='green', linestyle='--', linewidth=1)
    ax_bpm.axhline(120, color='blue', linestyle=':', linewidth=1.5, label='Bradykardia (<120)')
    ax_bpm.axhline(200, color='red', linestyle=':', linewidth=1.5, label='Tachykardia (>200)')
    ax_bpm.plot(bpm_df['time'], bpm_df['bpm'], marker='o', linestyle='-', color='black', label="Tętno (BPM)")
    ax_bpm.set_ylabel("Uderzenia na minutę (BPM)"); ax_bpm.set_xlabel("Czas")
    ax_bpm.grid(True, which='both', linestyle=':')
    handles, labels = ax_bpm.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax_bpm.legend(unique_labels.values(), unique_labels.keys())

    # Ustawienie formatowania osi X, aby pokazywała tylko godzinę i minuty
    formatter = mdates.DateFormatter('%H:%M') # Format: Godzina:Minuta
    ax_bpm.xaxis.set_major_formatter(formatter)
    
    # === TUTAJ DRUGA POPRAWKA ===
    # Wywołujemy metodę na przechwyconym obiekcie figury 'fig'
    fig.autofmt_xdate()

    # plt.tight_layout(); 
    plt.show()
else:
    print("Nie udało się obliczyć tętna dla żadnego bloku.")



if last_block_data_for_plot:
    print("\nGenerowanie wykresu diagnostycznego dla ostatniego zaakceptowanego bloku...")
    data = last_block_data_for_plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=False)
    fig.suptitle(f"Szczegółowa Analiza Ostatniego Bloku", fontsize=16)
    time_axis_ppg = np.arange(len(data['ppg_before_correction']))/SAMPLE_RATE_PPG
    time_axis_imu = np.arange(len(data['imu']))/SAMPLE_RATE_IMU
    ax1.set_title("Sygnał PPG")
    ax1.plot(time_axis_ppg, data['ppg_before_correction'], color='silver', label='Sygnał przed korektą skoku DC')
    ax1.plot(time_axis_ppg, data['ppg_final'], color='darkred', label='Sygnał finalny (po korekcie i wygładzeniu)', linewidth=1.5)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.7, label='Poziom zerowy')
    fill_min, fill_max = np.nanmin(data['ppg_final']), np.nanmax(data['ppg_final'])
    ax1.fill_between(time_axis_ppg, fill_min if np.isfinite(fill_min) else -1, fill_max if np.isfinite(fill_max) else 1, where=~data['mask'], color='red', alpha=0.3, label='Ruch (odrzucone)')
    ax1.legend(); ax1.grid(True); ax1.set_ylabel("Wartość"); ax1.set_xlabel("Czas [s]")
    ax1.set_xlim(left=0, right=max(time_axis_ppg, default=0))
    ax2.set_title("Sygnał Akcelerometru i Wariancja")
    ax2.plot(time_axis_imu, data['imu'].iloc[:, 3:6], alpha=0.7); ax2.legend(['AccX', 'AccY', 'AccZ'])
    ax2.set_ylabel("Wartość Akcelerometru"); ax2.set_xlabel("Czas [s]"); ax2.grid(True)
    ax2.set_xlim(left=0, right=max(time_axis_imu, default=0))
    ax2_var = ax2.twinx(); ax2_var.plot(time_axis_imu, data['variance'], color='purple', linestyle='--', label='Wariancja')
    ax2_var.axhline(VARIANCE_THRESHOLD, color='fuchsia', linestyle=':', label='Próg')
    ax2_var.legend(loc='lower right'); ax2_var.set_ylabel("Wariancja", color='purple')
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()