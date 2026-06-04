import warnings
warnings.filterwarnings("ignore")

import soundfile as sf
from scipy.signal import convolve
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.restoration import cycle_spin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SAMPLE_RATE = 44100
NAME_ORIGINAL_WAV = "./Sounds/Sound_44100[Hz]_2[byte].wav"


def wavelet_denoiser(signal, level=5, mode="hard", wavelet="db4"):
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(signal.size))

    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=mode)
        for c in coeffs[1:]
    ]

    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    return denoised_signal[:len(signal)]


def gaussian_kernel(size, sigma):
    x = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def plot_metric(
    metric_name,
    snr_values,
    wt_mean,
    wt_list,
    g_mean,
    g_list,
    filename
):
    snr_scatter_wt = []
    metric_scatter_wt = []
    snr_scatter_g = []
    metric_scatter_g = []

    for snr, metric_list in zip(snr_values, wt_list):
        snr_scatter_wt.extend([snr] * len(metric_list))
        metric_scatter_wt.extend(metric_list)

    for snr, metric_list in zip(snr_values, g_list):
        snr_scatter_g.extend([snr] * len(metric_list))
        metric_scatter_g.extend(metric_list)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        snr_scatter_wt,
        metric_scatter_wt,
        alpha=0.05,
        label=f"Окремі значення {metric_name} WT"
    )
    axes[0].plot(
        snr_values,
        wt_mean,
        linewidth=2,
        label=f"Середнє {metric_name} WT"
    )
    axes[0].scatter(
        snr_scatter_g,
        metric_scatter_g,
        alpha=0.05,
        label=f"Окремі значення {metric_name} GF"
    )
    axes[0].plot(
        snr_values,
        g_mean,
        linewidth=2,
        label=f"Середнє {metric_name} GF"
    )

    axes[0].set_xticks(np.arange(-10, 21, 2))
    axes[0].set_xlabel("SNR (дБ)")
    axes[0].set_ylabel(metric_name)
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_title("Лінійний масштаб")

    axes[1].scatter(
        snr_scatter_wt,
        metric_scatter_wt,
        alpha=0.05,
        label=f"Окремі значення {metric_name} WT"
    )
    axes[1].plot(
        snr_values,
        wt_mean,
        linewidth=2,
        label=f"Середнє {metric_name} WT"
    )
    axes[1].scatter(
        snr_scatter_g,
        metric_scatter_g,
        alpha=0.05,
        label=f"Окремі значення {metric_name} GF"
    )
    axes[1].plot(
        snr_values,
        g_mean,
        linewidth=2,
        label=f"Середнє {metric_name} GF"
    )

    axes[1].set_xticks(np.arange(-10, 21, 1))
    axes[1].set_xlabel("SNR (дБ)")
    axes[1].set_ylabel(metric_name)
    axes[1].grid(True)

    if metric_name != "R2":
        axes[1].set_yscale("log")

    axes[1].legend()
    axes[1].set_title("Логарифмічний масштаб")

    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()


def filtration_efficiency():
    data, fs_original = sf.read(NAME_ORIGINAL_WAV)

    if len(data.shape) > 1:
        data = data[:, 0]

    signal_power = np.mean(data ** 2)
    max_shifts = 5

    snr_values = []

    mse_wt_mean = []
    mse_wt_list = []
    mse_g_mean = []
    mse_g_list = []

    mae_wt_mean = []
    mae_wt_list = []
    mae_g_mean = []
    mae_g_list = []

    rmse_wt_mean = []
    rmse_wt_list = []
    rmse_g_mean = []
    rmse_g_list = []

    r2_wt_mean = []
    r2_wt_list = []
    r2_g_mean = []
    r2_g_list = []

    d_wt_mean = []
    d_wt_list = []
    d_g_mean = []
    d_g_list = []

    for snr_db in np.arange(-10, 21, 0.5):
        mse_wt = []
        mse_g = []

        mae_wt = []
        mae_g = []

        rmse_wt = []
        rmse_g = []

        r2_wt = []
        r2_g = []

        d_wt = []
        d_g = []

        for _ in range(10):
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(
                0,
                np.sqrt(noise_power),
                size=data.shape
            )

            noisy_signal = data + noise

            sig_filtered_wavelet = cycle_spin(
                noisy_signal,
                func=wavelet_denoiser,
                max_shifts=max_shifts,
                shift_steps=5,
                num_workers=1
            )

            kernel = gaussian_kernel(size=11, sigma=2)
            sig_filtered_gaussian = convolve(
                noisy_signal,
                kernel,
                mode="same"
            )

            mse_wavelet = mean_squared_error(data, sig_filtered_wavelet)
            mse_gaussian = mean_squared_error(data, sig_filtered_gaussian)

            mae_wavelet = mean_absolute_error(data, sig_filtered_wavelet)
            mae_gaussian = mean_absolute_error(data, sig_filtered_gaussian)

            rmse_wavelet = np.sqrt(mse_wavelet)
            rmse_gaussian = np.sqrt(mse_gaussian)

            r2_wavelet = r2_score(data, sig_filtered_wavelet)
            r2_gaussian = r2_score(data, sig_filtered_gaussian)

            d_wavelet = np.var(data - sig_filtered_wavelet)
            d_gaussian = np.var(data - sig_filtered_gaussian)

            mse_wt.append(mse_wavelet)
            mse_g.append(mse_gaussian)

            mae_wt.append(mae_wavelet)
            mae_g.append(mae_gaussian)

            rmse_wt.append(rmse_wavelet)
            rmse_g.append(rmse_gaussian)

            r2_wt.append(r2_wavelet)
            r2_g.append(r2_gaussian)

            d_wt.append(d_wavelet)
            d_g.append(d_gaussian)

        snr_values.append(snr_db)

        mse_wt_mean.append(np.mean(mse_wt))
        mse_wt_list.append(list(mse_wt))
        mse_g_mean.append(np.mean(mse_g))
        mse_g_list.append(list(mse_g))

        mae_wt_mean.append(np.mean(mae_wt))
        mae_wt_list.append(list(mae_wt))
        mae_g_mean.append(np.mean(mae_g))
        mae_g_list.append(list(mae_g))

        rmse_wt_mean.append(np.mean(rmse_wt))
        rmse_wt_list.append(list(rmse_wt))
        rmse_g_mean.append(np.mean(rmse_g))
        rmse_g_list.append(list(rmse_g))

        r2_wt_mean.append(np.mean(r2_wt))
        r2_wt_list.append(list(r2_wt))
        r2_g_mean.append(np.mean(r2_g))
        r2_g_list.append(list(r2_g))

        d_wt_mean.append(np.mean(d_wt))
        d_wt_list.append(list(d_wt))
        d_g_mean.append(np.mean(d_g))
        d_g_list.append(list(d_g))

    plot_metric(
        "MSE",
        snr_values,
        mse_wt_mean,
        mse_wt_list,
        mse_g_mean,
        mse_g_list,
        "./Sounds/MSE_vs_SNR.png"
    )

    plot_metric(
        "MAE",
        snr_values,
        mae_wt_mean,
        mae_wt_list,
        mae_g_mean,
        mae_g_list,
        "./Sounds/MAE_vs_SNR.png"
    )

    plot_metric(
        "RMSE",
        snr_values,
        rmse_wt_mean,
        rmse_wt_list,
        rmse_g_mean,
        rmse_g_list,
        "./Sounds/RMSE_vs_SNR.png"
    )

    plot_metric(
        "R2",
        snr_values,
        r2_wt_mean,
        r2_wt_list,
        r2_g_mean,
        r2_g_list,
        "./Sounds/R2_vs_SNR.png"
    )

    plot_metric(
        "D",
        snr_values,
        d_wt_mean,
        d_wt_list,
        d_g_mean,
        d_g_list,
        "./Sounds/D_vs_SNR.png"
    )

    print("Практична робота №6 виконана.")


if __name__ == "__main__":
    filtration_efficiency()