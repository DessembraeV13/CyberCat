import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
from scipy.signal import firwin, lfilter
import numpy as np
import os

def extract_number(filename):
    basename = os.path.basename(filename)  # np. "IMU123.csv"
    match = re.search(r'IMU(\d+)\.csv$', basename)
    return int(match.group(1)) if match else float('inf')

# Pobierz i posortuj pliki
file_list = sorted(glob.glob("IMU/*.csv"), key=extract_number)



# Wczytaj dane i zapamiętaj długości
data_frames = []
file_lengths = []
file_names = []

for file in file_list:
    df = pd.read_csv(file)
    data_frames.append(df)
    file_lengths.append(len(df))
    # file_names.append(file.split("/")[-1])  # tylko nazwa pliku
    file_names.append(os.path.basename(file))

# Punkty podziału danych
start_indices = np.cumsum([0] + file_lengths[:-1])

# Połączenie danych
data = pd.concat(data_frames, ignore_index=True)

# Filtr FIR
fs = 50
fc = 10
numtaps = 51
fir_coeff = firwin(numtaps, fc, fs=fs, pass_zero=True)

def apply_fir_filter(signal):
    return lfilter(fir_coeff, 1.0, signal)

# Filtrowanie
filtered_data = data.copy()
for col in ["Yaw", "Pitch", "Roll", "AX", "AY", "AZ", "GX", "GY", "GZ"]:
    filtered_data[col] = apply_fir_filter(data[col])

# Wykresy
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

# Rysowanie sygnałów
axes[0].plot(filtered_data["Yaw"], label="Yaw", color="b")
axes[0].plot(filtered_data["Pitch"], label="Pitch", color="g")
axes[0].plot(filtered_data["Roll"], label="Roll", color="r")
axes[0].set_ylabel("Kąt (stopnie)")
axes[0].set_title("Yaw, Pitch, Roll - Po filtracji FIR")
axes[0].legend()
axes[0].grid()

axes[1].plot(filtered_data["AX"], label="X", color="c")
axes[1].plot(filtered_data["AY"], label="Y", color="m")
axes[1].plot(filtered_data["AZ"], label="Z", color="y")
axes[1].set_ylabel("Przyspieszenie (m/s²)")
axes[1].set_title("AX, AY, AZ - Po filtracji FIR")
axes[1].legend()
axes[1].grid()

axes[2].plot(filtered_data["GX"], label="X", color="c")
axes[2].plot(filtered_data["GY"], label="Y", color="m")
axes[2].plot(filtered_data["GZ"], label="Z", color="y")
axes[2].set_ylabel("Przysp. kątowe (rad/s²)")
axes[2].set_title("GX, GY, GZ - Po filtracji FIR")
axes[2].legend()
axes[2].grid()

# Dodanie linii i opisów
for i, start in enumerate(start_indices):
    for ax in axes:
        # ax.axvline(start, color='k', linestyle='--', alpha=0.3)
        ax.text(start + 5, ax.get_ylim()[1]*0.9, file_names[i], rotation=90,
                verticalalignment='top', fontsize=8, color='black')

plt.xlabel("Numer próbki")
plt.tight_layout()
plt.show()
