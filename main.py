import warnings
warnings.filterwarnings("ignore")

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import pywt

from skimage.restoration import (
    denoise_wavelet,
    denoise_invariant,
    denoise_tv_chambolle,
    denoise_bilateral
)

SAMPLE_RATE = 44100
NAME_ORIGINAL_WAV = "./Sounds/Sound_44100[Hz]_2[byte].wav"


def wavelet_denoiser(signal, level, mode, wavelet):
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(signal.size))

    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=mode)
        for c in coeffs[1:]
    ]

    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    return denoised_signal[:len(signal)]


def invarince_denoiser(image, **kwargs):
    return denoise_wavelet(
        image,
        sigma=0.5,
        wavelet="db4",
        mode="soft"
    )


def save_plot(time, original, filtered, title, filename):
    plt.figure(figsize=(10, 6))

    plt.plot(time, original, 'b-', label='Original Clean Signal')
    plt.plot(time, filtered, 'g-', linewidth=2, label=title)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.close()


def sound_filter():

    data, fs_original = sf.read(NAME_ORIGINAL_WAV)

    if len(data.shape) > 1:
        data = data[:, 0]

    time = np.arange(len(data)) / fs_original

    data_2d = data.reshape(1, -1)

    invariance = denoise_invariant(
        data_2d,
        denoise_function=invarince_denoiser
    ).flatten()

    total_variation = denoise_tv_chambolle(
        data_2d,
        weight=0.1,
        channel_axis=None
    ).flatten()

    bilateral = denoise_bilateral(
        data_2d,
        sigma_color=0.05,
        sigma_spatial=15,
        channel_axis=None
    ).flatten()

    wavelet = wavelet_denoiser(
        data,
        level=5,
        mode="soft",
        wavelet="db4"
    )

    sf.write(
        "./Sounds/Filtered_Invariance.wav",
        invariance,
        SAMPLE_RATE
    )

    sf.write(
        "./Sounds/Filtered_Total_Variation.wav",
        total_variation,
        SAMPLE_RATE
    )

    sf.write(
        "./Sounds/Filtered_Bilateral.wav",
        bilateral,
        SAMPLE_RATE
    )

    sf.write(
        "./Sounds/Filtered_Wavelet.wav",
        wavelet,
        SAMPLE_RATE
    )

    save_plot(
        time,
        data,
        invariance,
        "J-Invariance",
        "./Sounds/J_Invariance.png"
    )

    save_plot(
        time,
        data,
        total_variation,
        "Total Variation",
        "./Sounds/Total_Variation.png"
    )

    save_plot(
        time,
        data,
        bilateral,
        "Bilateral",
        "./Sounds/Bilateral.png"
    )

    save_plot(
        time,
        data,
        wavelet,
        "Wavelet",
        "./Sounds/Wavelet.png"
    )

    print("Практична робота №3 виконана.")


if __name__ == "__main__":
    sound_filter()