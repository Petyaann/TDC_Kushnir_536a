import warnings
warnings.filterwarnings("ignore")

import soundfile as sf
from scipy.signal import resample
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob

SAMPLE_RATE = 44100

NAME_ORIGINAL_WAV = "./Sounds/Sound_44100[Hz]_2[byte].wav"
NAME_RESAMPLED_WAV = "./Sounds/Sound_4000[Hz]_2[byte].wav"


def to_scientific_pretty(x, precision=2):
    superscripts = str.maketrans(
        "0123456789-",
        "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"
    )

    mantissa, exponent = f"{x:.{precision}e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")

    return f"{mantissa} · 10{str(int(exponent)).translate(superscripts)}"


def make_metrics_table():
    results = []
    row = []
    headers = ["MSE", "MAE", "RMSE", "R2", "D"]

    data_original, fs_original = sf.read(NAME_ORIGINAL_WAV)

    if len(data_original.shape) > 1:
        data_original = data_original[:, 0]

    wav_files = glob.glob("./Sounds/*.wav")

    for sounds in wav_files:
        sounds = sounds.replace("\\", "/")

        if sounds == NAME_ORIGINAL_WAV:
            continue

        if sounds == NAME_RESAMPLED_WAV:
            row.append("Ресемпл 4 кГц")

            data, fs = sf.read(sounds)

            if len(data.shape) > 1:
                data = data[:, 0]

            data = resample(data, len(data_original))

        else:
            type_filter = sounds.replace("./Sounds/Filtered_", "")
            type_filter = type_filter.replace(".wav", "")
            type_filter = type_filter.replace("_", " ")

            if type_filter == "4000[Hz] 2[byte]":
                type_filter = "Лінійний фільтр 4 кГц"

            row.append(type_filter)

            data, fs = sf.read(sounds)

            if len(data.shape) > 1:
                data = data[:, 0]

            if len(data) != len(data_original):
                data = resample(data, len(data_original))

        mse = mean_squared_error(data_original, data)
        mae = mean_absolute_error(data_original, data)
        rmse = np.sqrt(mse)
        r2 = r2_score(data_original, data)
        d = np.var(data_original - data)

        results.append([
            to_scientific_pretty(mse),
            to_scientific_pretty(mae),
            to_scientific_pretty(rmse),
            round(r2, 2),
            to_scientific_pretty(d)
        ])

    n_rows = len(row)
    n_cols = len(headers)

    fig, ax = plt.subplots(figsize=(n_cols * 2.8, n_rows * 0.4))
    ax.axis("off")

    table = ax.table(
        cellText=results,
        rowLabels=row,
        colLabels=headers,
        cellLoc="center",
        rowLoc="center",
        bbox=[0.08, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    plt.tight_layout()
    plt.savefig("./Sounds/Metrics_Table.png", dpi=600)
    plt.show()

    print("Практична робота №5 виконана.")


if __name__ == "__main__":
    make_metrics_table()