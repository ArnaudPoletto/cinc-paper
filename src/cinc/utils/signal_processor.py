"""
This utility file provides functions for processing signals, including filtering, resampling, and normalization.

Authors: Arnaud Poletto
"""

import numpy as np
from scipy import signal
from scipy.stats import median_abs_deviation


class SignalProcessor:
    """
    A class providing static methods for various signal processing tasks such as filtering, resampling, and normalization.
    """

    @staticmethod
    def powerline_filter(
        signal_data: np.ndarray,
        fs: float,
        powerline_freq: float = 50.0,
        quality_factor: float = 30.0,
        remove_harmonics: bool = True,
        max_harmonic: int = 4,
    ) -> np.ndarray:
        """
        Apply a notch filter to remove powerline interference and its harmonics from the signal.

        Args:
            signal_data (np.ndarray): The input signal data.
            fs (float): The sampling frequency of the signal.
            powerline_freq (float, optional): The fundamental powerline frequency to remove. Defaults to 50.0 Hz.
            quality_factor (float, optional): The quality factor for the notch filter. Defaults to 30.0.
            remove_harmonics (bool, optional): Whether to remove harmonic frequencies. Defaults to True.
            max_harmonic (int, optional): The maximum harmonic order to remove. Defaults to 4.

        Returns:
            filtered_signal_data (np.ndarray): The filtered signal data.
        """
        filtered_signal_data = signal_data.copy()
        nyquist = fs / 2

        # Remove fundamental frequency
        if powerline_freq < nyquist:
            b, a = signal.iirnotch(powerline_freq, quality_factor, fs)
            filtered_signal_data = signal.filtfilt(b, a, filtered_signal_data)

        # Remove harmonic frequencies if requested
        if remove_harmonics:
            for harmonic_order in range(2, max_harmonic + 1):
                harmonic_freq = powerline_freq * harmonic_order
                if harmonic_freq < nyquist:
                    b, a = signal.iirnotch(harmonic_freq, quality_factor, fs)
                    filtered_signal_data = signal.filtfilt(b, a, filtered_signal_data)

        return filtered_signal_data

    @staticmethod
    def bandpass_filter(
        signal_data: np.ndarray,
        lowcut: float,
        highcut: float,
        fs: float,
        order: int,
    ) -> np.ndarray:
        """
        Apply a Butterworth bandpass filter to the signal.

        Args:
            signal_data (np.ndarray): The input signal data.
            lowcut (float): The low cutoff frequency of the bandpass filter.
            highcut (float): The high cutoff frequency of the bandpass filter.
            fs (float): The sampling frequency of the signal.
            order (int): The order of the Butterworth filter.

        Returns:
            filtered_signal_data (np.ndarray): The filtered signal data.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        # Check for problematic normalized frequencies that can cause instability
        if low < 0.001 or high > 0.99 or (high - low) < 0.001:
            # Use second-order sections (SOS) for better numerical stability
            sos = signal.butter(order, [low, high], btype="band", output="sos")
            filtered_signal_data = signal.sosfiltfilt(sos, signal_data)
        else:
            # Use traditional design for normal parameters
            b, a = signal.butter(order, [low, high], btype="band")
            filtered_signal_data = signal.filtfilt(b, a, signal_data)

        return filtered_signal_data

    @staticmethod
    def lowpass_filter(
        signal_data: np.ndarray,
        highcut: float,
        fs: float,
        order: int,
    ) -> np.ndarray:
        """
        Apply a Butterworth lowpass filter to the signal.

        Args:
            signal_data (np.ndarray): The input signal data.
            highcut (float): The cutoff frequency of the lowpass filter.
            fs (float): The sampling frequency of the signal.
            order (int): The order of the Butterworth filter.

        Returns:
            filtered_signal_data (np.ndarray): The filtered signal data.
        """
        nyquist = 0.5 * fs
        high = highcut / nyquist

        # Check for problematic normalized frequencies
        if high > 0.99:
            sos = signal.butter(order, high, btype="low", output="sos")
            filtered_signal_data = signal.sosfiltfilt(sos, signal_data)
        else:
            b, a = signal.butter(order, high, btype="low")
            filtered_signal_data = signal.filtfilt(b, a, signal_data)

        return filtered_signal_data

    @staticmethod
    def resample_signal(
        signal_data: np.ndarray, original_fs: float, target_fs: float
    ) -> np.ndarray:
        """
        Resample the signal to a target sampling frequency.

        Args:
            signal_data (np.ndarray): The input signal data.
            original_fs (float): The original sampling frequency of the signal.
            target_fs (float): The desired target sampling frequency.

        Returns:
            resampled_signal_data (np.ndarray): The resampled signal data.
        """
        num_samples = int(len(signal_data) * target_fs / original_fs)
        return signal.resample(signal_data, num_samples)

    @staticmethod
    def normalize_signal(
        signal_data: np.ndarray,
        method: str,
    ) -> np.ndarray:
        """
        Normalize the signal using the specified method.

        Args:
            signal_data (np.ndarray): The input signal data.
            method (str): The normalization method to use. Options are 'zscore' or 'robust'.

        Returns:
            normalized_signal_data (np.ndarray): The normalized signal data.
        """
        if method not in ["zscore", "robust"]:
            print(
                f"⚠️  Unknown normalization method '{method}', returning original signal"
            )
            return signal_data

        if method == "zscore":
            # Use nanmean and nanstd to handle any NaN values
            mean_val = np.nanmean(signal_data)
            std_val = np.nanstd(signal_data)

            # Check for problematic values
            if np.isnan(mean_val) or np.isnan(std_val):
                print(
                    f"⚠️  NaN detected in signal statistics (mean={mean_val}, std={std_val}), returning original signal"
                )
                return signal_data

            if std_val == 0 or np.isinf(std_val):
                print(
                    f"⚠️  Invalid std deviation ({std_val}), returning mean-centered signal"
                )
                return signal_data - mean_val

            return (signal_data - mean_val) / std_val

        if method == "robust":
            percentile_low = np.nanpercentile(signal_data, 0.1)
            percentile_high = np.nanpercentile(signal_data, 99.9)
            signal_clipped = np.clip(signal_data, percentile_low, percentile_high)

            # Use nanmedian and MAD to handle any NaN values
            median_val = np.nanmedian(signal_clipped)
            mad_val = median_abs_deviation(
                signal_clipped, nan_policy="omit", scale="normal"
            )

            # Check for problematic values
            if np.isnan(median_val) or np.isnan(mad_val):
                print(
                    f"⚠️  NaN detected in signal statistics (median={median_val}, MAD={mad_val}), returning original signal"
                )
                return signal_data

            if mad_val == 0 or np.isinf(mad_val):
                print(f"⚠️  Invalid MAD ({mad_val}), returning median-centered signal")
                return signal_data - median_val

            return (signal_data - median_val) / mad_val
