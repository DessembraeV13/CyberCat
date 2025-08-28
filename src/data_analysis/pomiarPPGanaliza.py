import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, medfilt, correlate
import heartpy as hp
import os
import joblib
import re
from scipy.ndimage import binary_dilation
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq

# --- KONFIGURACJA GŁÓWNA ---
IMU_FOLDER = "futro"
PPG_FOLDER = "futro"
MODEL_FILENAME = "activity_classifier_model.joblib"
START_PPG_BLOCK_INDEX = 12 
SAMPLE_RATE_PPG, SAMPLE_RATE_IMU = 200, 50
SEGMENT_LENGTH_IMU = 250
MOTION_LABEL = 'ruch'
LPF_CUTOFF, HPF_CUTOFF, NUM_TAPS = 6, 1, 201
VARIANCE_WINDOW_SIZE, VARIANCE_THRESHOLD = int(0.5 * SAMPLE_RATE_IMU), 100000
SAFETY_MARGIN_SECONDS_PPG = 3
INITIAL_TRIM_SECONDS = 15 
REJECT_RATIO_THRESHOLD = 0.4
# === NOWA KONFIGURACJA: Filtr wygładzający ===
SMOOTHING_LPF_CUTOFF = 10.0 # Częstotliwość odcięcia (Hz)
SMOOTHING_NUM_TAPS = 51    # Mniejsza liczba "taps" dla delikatniejszego efektu

# --- FUNKCJE POMOCNICZE (bez zmian) ---
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
    if errors_found > 0: print(f"  - Naprawiono {errors_found} wierszy IMU.")
    return df_fixed
def extract_features(segment):
    features = []
    for i in range(segment.shape[1]):
        axis_data = segment[:, i]
        features.extend([np.mean(axis_data), np.std(axis_data), np.var(axis_data), np.min(axis_data), np.max(axis_data), np.median(axis_data), skew(axis_data), kurtosis(axis_data), np.quantile(axis_data, 0.75) - np.quantile(axis_data, 0.25), np.sum(np.square(axis_data))])
        N, yf, xf = len(axis_data), rfft(axis_data), rfftfreq(len(axis_data), 1 / SAMPLE_RATE_IMU)
        if len(yf) > 1:
            dominant_freq_idx = np.argmax(np.abs(yf[1:])) + 1
            features.extend([xf[dominant_freq_idx], np.abs(yf[dominant_freq_idx])])
        else: features.extend([0, 0])
    return features
# --- GŁÓWNA LOGIKA SKRYPTU ---

# === KROK 1 & 2 (bez zmian) ===
print(f"Krok 1: Wczytywanie bloku pomiarowego PPG od indeksu {START_PPG_BLOCK_INDEX}...")
imu_block_start_index = (START_PPG_BLOCK_INDEX // 4) * 10
print(f"  Logika mapowania: Blok PPG od {START_PPG_BLOCK_INDEX} -> Blok IMU od {imu_block_start_index}")
block_imu_dfs, block_ppg_dfs, valid_block = [], [], True
for i in range(4):
    ppg_file_num, imu_file_num = START_PPG_BLOCK_INDEX + i, imu_block_start_index + i
    ppg_path, imu_path = os.path.join(PPG_FOLDER, f"PPG{ppg_file_num}.csv"), os.path.join(IMU_FOLDER, f"IMU{imu_file_num}.csv")
    if os.path.exists(ppg_path) and os.path.exists(imu_path):
        print(f"    - Parowanie: {os.path.basename(ppg_path)} -> {os.path.basename(imu_path)}")
        df_ppg = pd.read_csv(ppg_path, na_values=['-']).interpolate(method='linear', limit_direction='both')
        df_imu = pd.read_csv(imu_path, na_values=['-']).interpolate(method='linear', limit_direction='both')
        df_imu = apply_median_filter(df_imu)
        min_duration_sec = min(len(df_ppg)/SAMPLE_RATE_PPG, len(df_imu)/SAMPLE_RATE_IMU)
        samples_ppg, samples_imu = int(min_duration_sec*SAMPLE_RATE_PPG), int(min_duration_sec*SAMPLE_RATE_IMU)
        block_ppg_dfs.append(df_ppg.head(samples_ppg)); block_imu_dfs.append(napraw_bledy_IMU(df_imu.head(samples_imu)))
    else: print(f"BŁĄD: Brak pary plików dla PPG{ppg_file_num}/IMU{imu_file_num}."); valid_block = False; break
if not valid_block: exit()
block_ppg_df, block_imu_df = pd.concat(block_ppg_dfs, ignore_index=True), pd.concat(block_imu_dfs, ignore_index=True)
trim_samples_ppg, trim_samples_imu = int(INITIAL_TRIM_SECONDS * SAMPLE_RATE_PPG), int(INITIAL_TRIM_SECONDS * SAMPLE_RATE_IMU)
if len(block_ppg_df) > trim_samples_ppg and len(block_imu_df) > trim_samples_imu:
    block_ppg_df, block_imu_df = block_ppg_df.iloc[trim_samples_ppg:].reset_index(drop=True), block_imu_df.iloc[trim_samples_imu:].reset_index(drop=True)
    print(f"  Przycięto początkowe {INITIAL_TRIM_SECONDS}s sygnału.")
print("\nKrok 2: Hybrydowe wykrywanie ruchu...")
try: model = joblib.load(MODEL_FILENAME)
except FileNotFoundError: print(f"BŁĄD: Nie znaleziono pliku modelu '{MODEL_FILENAME}'."); exit()
motion_mask_ai = np.ones(len(block_imu_df), dtype=bool)
for i in range(0, len(block_imu_df) - SEGMENT_LENGTH_IMU + 1, SEGMENT_LENGTH_IMU):
    if model.predict([extract_features(block_imu_df.iloc[i:i+SEGMENT_LENGTH_IMU].values)])[0] == MOTION_LABEL:
        motion_mask_ai[i : i + SEGMENT_LENGTH_IMU] = False
rolling_variance = block_imu_df.iloc[:, 3:6].rolling(window=VARIANCE_WINDOW_SIZE).var().sum(axis=1).fillna(0)
final_motion_mask_imu = np.logical_and(motion_mask_ai, rolling_variance < VARIANCE_THRESHOLD)

# === KROK 3: Filtracja i przygotowanie sygnału PPG ===
print("\nKrok 3: Filtracja i przygotowanie sygnału PPG...")
ir_signal = block_ppg_df["IR"].values

# === NOWY KOD: Obliczanie składowej DC ===
# Obliczamy składową DC jako średnią wartość surowego sygnału IR.
# Jest to najlepszy moment, przed usunięciem tej informacji przez filtry pasmowo-przepustowe.
dc_component = np.mean(ir_signal)
print(f"  Obliczono składową DC z surowego sygnału: {dc_component:.2f}")
# === KONIEC NOWEGO KODU ===

filtered_ir = filtfilt(firwin(NUM_TAPS, LPF_CUTOFF, fs=SAMPLE_RATE_PPG), 1.0, ir_signal)
filtered_ir = filtfilt(firwin(NUM_TAPS, HPF_CUTOFF, fs=SAMPLE_RATE_PPG, pass_zero="highpass"), 1.0, filtered_ir)
# filtered_ir = hp.flip_signal(filtered_ir)
detrended_ir = filtered_ir - np.mean(filtered_ir)
signal_before_correction = detrended_ir.copy()
print("  Uruchamianie dynamicznej synchronizacji sygnałów...")
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
delay_seconds = delay_samples_ppg / SAMPLE_RATE_PPG
if delay_seconds > 0: print(f"  Synchronizacja: Sygnał PPG jest opóźniony o {delay_seconds:.3f} s.")
else: print(f"  Synchronizacja: Sygnał IMU jest opóźniony o {-delay_seconds:.3f} s.")
shifted_mask_imu = np.roll(final_motion_mask_imu, delay_samples_imu)
motion_mask_ppg_narrow = np.repeat(shifted_mask_imu, resampling_factor)
target_len, current_len = len(detrended_ir), len(motion_mask_ppg_narrow)
if current_len > target_len: motion_mask_ppg_narrow = motion_mask_ppg_narrow[:target_len]
elif current_len < target_len: motion_mask_ppg_narrow = np.pad(motion_mask_ppg_narrow, (0, target_len - current_len), mode='edge')
safety_margin_samples = int(SAFETY_MARGIN_SECONDS_PPG * SAMPLE_RATE_PPG)
dilated_motion_areas = binary_dilation(~motion_mask_ppg_narrow, structure=np.ones(safety_margin_samples))
final_motion_mask_ppg = ~dilated_motion_areas
print("  Korekta skoku linii bazowej po artefakcie...")
corrected_ir = detrended_ir.copy()
motion_indices = np.where(~final_motion_mask_ppg)[0]
if len(motion_indices) > 0:
    motion_start_idx, motion_end_idx = motion_indices[0], motion_indices[-1]
    stable_window_samples = int(1.0 * SAMPLE_RATE_PPG)
    if motion_start_idx > stable_window_samples and motion_end_idx < len(detrended_ir) - stable_window_samples:
        mean_before = np.mean(detrended_ir[motion_start_idx - stable_window_samples : motion_start_idx])
        mean_after = np.mean(detrended_ir[motion_end_idx + 1 : motion_end_idx + 1 + stable_window_samples])
        dc_offset = mean_after - mean_before
        print(f"    Wykryto skok DC o wartości {dc_offset:.2f}. Korygowanie sygnału.")
        corrected_ir[motion_end_idx + 1:] -= dc_offset
corrected_ir[~final_motion_mask_ppg] = np.nan
interpolated_ir = pd.Series(corrected_ir).interpolate(method='linear', limit_direction='both').bfill().ffill().values
print("  Ostateczne centrowanie sygnału...")
interpolated_ir = interpolated_ir - np.mean(interpolated_ir)

# === NOWY KROK: Końcowe wygładzanie sygnału ===
print("  Końcowe wygładzanie sygnału w celu usunięcia 'kanciastości'...")
final_signal = filtfilt(firwin(SMOOTHING_NUM_TAPS, SMOOTHING_LPF_CUTOFF, fs=SAMPLE_RATE_PPG), 1.0, interpolated_ir)

# Wklej ten kod w miejsce KROKU 4 i 5 w swoim skrypcie.
# Poprzednie kroki (1-3) pozostają bez zmian.

# === KROK 4: Analiza sygnału PPG z kontrolą jakości ===
print("\nKrok 4: Analiza sygnału PPG z kontrolą jakości...")
measures, working_data = {}, {}
ac_component, perfusion_index = np.nan, np.nan
is_good_quality = False # Domyślnie pomiar jest słabej jakości

try:
    # --- PRÓBA ANALIZY Z HEARTPY ---
    working_data, measures = hp.process(final_signal, SAMPLE_RATE_PPG, interp_threshold=1000, bpmmin=40, bpmmax=200)
    
    # Sprawdzanie, czy HeartPy dało sensowny wynik
    if 'bpm' in measures and pd.notna(measures['bpm']):
        num_rejected = len(working_data.get('removed_beats', []))
        num_accepted = len(working_data.get('peaklist', []))
        
        if (num_rejected + num_accepted) == 0:
            rejection_reason = "HeartPy nie znalazł żadnych pików."
        else:
            reject_ratio = num_rejected / (num_rejected + num_accepted)
            if reject_ratio > REJECT_RATIO_THRESHOLD:
                rejection_reason = f"Stosunek odrzuconych pików ({reject_ratio:.0%}) przekroczył próg."
            else:
                is_good_quality = True
                rejection_reason = ""

        if is_good_quality:
            print("\nAnaliza HeartPy zakończona sukcesem. Obliczanie metryk AC/PI na podstawie pików...")
            peak_indices = working_data.get('peaklist', [])
            beat_amplitudes = []
            if len(peak_indices) > 1:
                for i in range(len(peak_indices) - 1):
                    start_idx, end_idx = peak_indices[i], peak_indices[i+1]
                    peak_value = final_signal[start_idx]
                    trough_value = np.min(final_signal[start_idx:end_idx])
                    beat_amplitudes.append(peak_value - trough_value)
                
                if beat_amplitudes:
                    ac_component = np.mean(beat_amplitudes)
            
    else:
        rejection_reason = "HeartPy nie obliczyło BPM (wynik to 'nan')."

except Exception as e:
    rejection_reason = f"Błąd krytyczny w HeartPy: {e}"
    print(f"BŁĄD: {rejection_reason}")

# === NOWA LOGIKA: Metoda alternatywna, jeśli HeartPy zawiodło ===
if not is_good_quality:
    print(f"\n--- POMIAR ODRZUCONY PRZEZ HEARTPY ---\nPowód: {rejection_reason}")
    print("Uruchamianie alternatywnej metody oceny jakości sygnału (na podstawie odchylenia standardowego).")
    measures = {} 
    
    # === KLUCZOWA POPRAWKA ===
    # Oszacowanie "mocy" składowej AC jako odchylenia standardowego WYŁĄCZNIE STABILNYCH segmentów sygnału.
    # Używamy np.nanstd na sygnale `corrected_ir`, gdzie artefakty są oznaczone jako NaN.
    # To zapobiega zawyżaniu wyniku przez energię artefaktów ruchowych.
    ac_component = np.nanstd(corrected_ir)
    print(f"  Oszacowana moc AC (odch. standardowe stabilnych segmentów): {ac_component:.2f}")


# === ZAWSZE OBLICZAMY PI (na podstawie dokładnego lub szacowanego AC) ===
if dc_component > 0 and pd.notna(ac_component):
    perfusion_index = (ac_component / dc_component) * 100
    print(f"  Obliczono Indeks Perfuzji (PI): {perfusion_index:.3f}%")
else:
    perfusion_index = np.nan
    print("  OSTRZEŻENIE: Nie można obliczyć Indeksu Perfuzji (brak AC lub DC<=0).")

# === PODSUMOWANIE I ZAPISANIE WYNIKÓW ===
measures['dc_component'] = dc_component
measures['ac_component_method'] = 'peaks' if is_good_quality else 'std_dev_on_clean'
measures['ac_component'] = ac_component
measures['perfusion_index'] = perfusion_index

if is_good_quality:
    print("\n--- PODSUMOWANIE ANALIZY - POMIAR ZAAKCEPTOWANY ---")
else:
    print("\n--- PODSUMOWANIE ANALIZY - POMIAR ODRZUCONY (ale z metrykami jakości) ---")

for measure, value in measures.items(): 
    if value is not None and pd.notna(value):
        if isinstance(value, float):
            if measure == 'perfusion_index':
                print(f'{measure}: {value:.3f}%')
            else:
                print(f'{measure}: {value:.2f}')
        else:
            print(f'{measure}: {value}')
            
# Krok 5 (wykresy) pozostaje bez zmian


# === KROK 5: Generowanie wykresu diagnostycznego ===
print("\nKrok 5: Generowanie wykresu diagnostycznego...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=False)

# === ZMODYFIKOWANY KOD: Dynamiczny tytuł główny z metrykami (działa zawsze) ===
main_title = f"Analiza PPG - Lokalizacja: {PPG_FOLDER.upper()} (Plik: PPG{START_PPG_BLOCK_INDEX})"
pi_text = f"PI: {perfusion_index:.2f}%" if pd.notna(perfusion_index) else "PI: N/A"

if is_good_quality:
    hr_text = f"HR: {measures.get('bpm'):.1f} bpm"
    main_title += f"\n[WYNIK: {hr_text} | {pi_text} | Jakość: DOBRA]"
else:
    main_title += f"\n[WYNIK: Tętno niewykrywalne | {pi_text} | Jakość: SŁABA]"

fig.suptitle(main_title, fontsize=16)
# === KONIEC ZMODYFIKOWANEGO KODU ===

time_axis_ppg = np.arange(len(signal_before_correction))/SAMPLE_RATE_PPG
time_axis_imu = np.arange(len(block_imu_df))/SAMPLE_RATE_IMU
ax1.set_title("Sygnał PPG (MAX30102)")
ax1.plot(time_axis_ppg, signal_before_correction, color='silver', label='Sygnał po filtracji pasmowej')
ax1.plot(time_axis_ppg, final_signal, color='darkred', label='Sygnał finalny (po korekcji i wygładzeniu)', linewidth=1.5)

# === ZMODYFIKOWANY KOD: Wizualizacja pików tylko jeśli pomiar jest dobry ===
if is_good_quality and 'peaklist' in working_data:
    peak_indices = working_data['peaklist']
    ax1.scatter(time_axis_ppg[peak_indices], final_signal[peak_indices], c='lime', s=60, label='Wykryte piki', zorder=5)
# === KONIEC ZMODYFIKOWANEGO KODU ===

ax1.axhline(0, color='black', linestyle='--', linewidth=0.7, label='Poziom zerowy')
fill_min, fill_max = np.nanmin(final_signal), np.nanmax(final_signal)
ax1.fill_between(time_axis_ppg, fill_min if np.isfinite(fill_min) else -1, fill_max if np.isfinite(fill_max) else 1, where=~final_motion_mask_ppg, color='red', alpha=0.3, label='Ruch (odrzucone)')
ax1.set_ylabel("Wartość"); ax1.set_xlabel("Czas [s]"); ax1.legend(); ax1.grid(True); ax1.set_xlim(left=0, right=max(time_axis_ppg, default=0))
ax2.set_title("Sygnał Akcelerometru (IMU) i Jego Wariancja")
ax2.plot(time_axis_imu, block_imu_df.iloc[:, 3:6], alpha=0.7); ax2.legend(['AccX', 'AccY', 'AccZ'])
ax2.set_ylabel("Wartość Akcelerometru"); ax2.set_xlabel("Czas [s]"); ax2.grid(True); ax2.set_xlim(left=0, right=max(time_axis_imu, default=0))
ax2_var = ax2.twinx()
ax2_var.plot(time_axis_imu, rolling_variance, color='purple', linestyle='--', label='Suma Wariancji')
ax2_var.axhline(VARIANCE_THRESHOLD, color='fuchsia', linestyle=':', label=f'Próg Wariancji ({VARIANCE_THRESHOLD})')
ax2_var.set_ylabel("Suma Wariancji", color='purple'); ax2_var.legend(loc='lower right')
plt.tight_layout(rect=[0, 0, 1, 0.94]) 
plt.show()

# ZMODYFIKOWANY KOD: Generuj wykresy HeartPy tylko dla dobrych pomiarów
if is_good_quality:
    print("\nGenerowanie standardowych wykresów HeartPy...");
    try:
        hp.plotter(working_data, measures); hp.plot_poincare(working_data, measures); plt.show()
    except Exception as e:
        print(f"Nie udało się wygenerować wykresów HeartPy: {e}")
# === KONIEC ZMODYFIKOWANEGO KODU ===

time_axis_ppg = np.arange(len(signal_before_correction))/SAMPLE_RATE_PPG
time_axis_imu = np.arange(len(block_imu_df))/SAMPLE_RATE_IMU
ax1.set_title("Sygnał PPG (MAX30102)")
ax1.plot(time_axis_ppg, signal_before_correction, color='silver', label='Sygnał po filtracji pasmowej')
ax1.plot(time_axis_ppg, final_signal, color='darkred', label='Sygnał finalny (po korekcji i wygładzeniu)', linewidth=1.5)

# === ZMODYFIKOWANY KOD: Wizualizacja pików, jeśli zostały znalezione ===
if 'peaklist' in working_data and measures:
    peak_indices = working_data['peaklist']
    ax1.scatter(time_axis_ppg[peak_indices], final_signal[peak_indices], c='lime', s=60, label='Wykryte piki', zorder=5)
# === KONIEC ZMODYFIKOWANEGO KODU ===

ax1.axhline(0, color='black', linestyle='--', linewidth=0.7, label='Poziom zerowy')
fill_min, fill_max = np.nanmin(final_signal), np.nanmax(final_signal)
ax1.fill_between(time_axis_ppg, fill_min if np.isfinite(fill_min) else -1, fill_max if np.isfinite(fill_max) else 1, where=~final_motion_mask_ppg, color='red', alpha=0.3, label='Ruch (odrzucone)')
ax1.set_ylabel("Wartość"); ax1.set_xlabel("Czas [s]"); ax1.legend(); ax1.grid(True); ax1.set_xlim(left=0, right=max(time_axis_ppg, default=0))
ax2.set_title("Sygnał Akcelerometru (IMU) i Jego Wariancja")
ax2.plot(time_axis_imu, block_imu_df.iloc[:, 3:6], alpha=0.7); ax2.legend(['AccX', 'AccY', 'AccZ'])
ax2.set_ylabel("Wartość Akcelerometru"); ax2.set_xlabel("Czas [s]"); ax2.grid(True); ax2.set_xlim(left=0, right=max(time_axis_imu, default=0))
ax2_var = ax2.twinx()
ax2_var.plot(time_axis_imu, rolling_variance, color='purple', linestyle='--', label='Suma Wariancji')
ax2_var.axhline(VARIANCE_THRESHOLD, color='fuchsia', linestyle=':', label=f'Próg Wariancji ({VARIANCE_THRESHOLD})')
ax2_var.set_ylabel("Suma Wariancji", color='purple'); ax2_var.legend(loc='lower right')
plt.tight_layout(rect=[0, 0, 1, 0.94]) # Dostosowano rect, by tytuł się zmieścił
plt.show()
if measures:
    print("\nGenerowanie standardowych wykresów HeartPy...");
    try:
        hp.plotter(working_data, measures); hp.plot_poincare(working_data, measures); plt.show()
    except Exception as e:
        print(f"Nie udało się wygenerować wykresów HeartPy: {e}")