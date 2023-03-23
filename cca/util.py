import math
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from scipy.signal import butter, filtfilt


def butter_bandpass_filter(data, lowcut, highcut, sample_rate, order):  # 巴特沃斯带通滤波,order-->滤波器阶数
    nyp = 0.5 * sample_rate
    low = lowcut / nyp
    high = highcut / nyp
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def get_filtered_eeg(eeg, lowcut, highcut, order, sample_rate):
    num_classes = eeg.shape[0]
    num_chan = eeg.shape[1]
    total_trial_len = eeg.shape[2]
    num_trials = eeg.shape[3]

    trial_len = int(38 + 0.135 * sample_rate + 4 * sample_rate - 1) - int(38 + 0.135 * sample_rate)
    filtered_data = np.zeros((eeg.shape[0], eeg.shape[1], trial_len, eeg.shape[3]))

    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                signal_to_filter = np.squeeze(eeg[target, channel, int(38 + 0.135 * sample_rate):
                                                                   int(38 + 0.135 * sample_rate + 4 * sample_rate - 1),
                                              trial])
                filtered_data[target, channel, :, trial] = butter_bandpass_filter(signal_to_filter, lowcut,
                                                                                  highcut, sample_rate, order)
    return filtered_data


def buffer(data, duration, data_overlap):
    number_segments = int(math.ceil((len(data) - data_overlap) / (duration - data_overlap)))
    temp_buf = [data[i:i + duration] for i in range(0, len(data), (duration - int(data_overlap)))]
    temp_buf[number_segments - 1] = np.pad(temp_buf[number_segments - 1],
                                           (0, duration - temp_buf[number_segments - 1].shape[0]),
                                           'constant')
    segmented_data = np.vstack(temp_buf[0:number_segments])

    return segmented_data


def get_segmented_epochs(data, window_len, shift_len, sample_rate):
    num_classes = data.shape[0]
    num_chan = data.shape[1]
    num_trials = data.shape[3]

    duration = int(window_len * sample_rate)
    data_overlap = (window_len - shift_len) * sample_rate

    number_of_segments = int(math.ceil((data.shape[2] - data_overlap) /
                                       (duration - data_overlap)))

    segmented_data = np.zeros((data.shape[0], data.shape[1],
                               data.shape[3], number_of_segments, duration))

    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                segmented_data[target, channel, trial, :, :] = buffer(data[target, channel, :, trial],
                                                                      duration, data_overlap)

    return segmented_data


def magnitude_spectrum_features(segmented_data, FFT_PARAMS):  # magnitude spectrum features of the input EEG

    num_classes = segmented_data.shape[0]
    num_chan = segmented_data.shape[1]
    num_trials = segmented_data.shape[2]
    number_of_segments = segmented_data.shape[3]
    fft_len = segmented_data[0, 0, 0, 0, :].shape[0]

    NFFT = round(FFT_PARAMS['sampling_rate'] / FFT_PARAMS['resolution'])
    fft_index_start = int(round(FFT_PARAMS['start_frequency'] / FFT_PARAMS['resolution']))
    fft_index_end = int(round(FFT_PARAMS['end_frequency'] / FFT_PARAMS['resolution'])) + 1

    features_data = np.zeros(((fft_index_end - fft_index_start),
                              segmented_data.shape[1], segmented_data.shape[0],
                              segmented_data.shape[2], segmented_data.shape[3]))

    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                for segment in range(0, number_of_segments):
                    temp_FFT = np.fft.fft(segmented_data[target, channel, trial, segment, :], NFFT) / fft_len
                    magnitude_spectrum = 2 * np.abs(temp_FFT)
                    features_data[:, channel, target, trial, segment] = magnitude_spectrum[
                                                                        fft_index_start:fft_index_end, ]

    return features_data


def complex_spectrum_features(segmented_data, FFT_PARAMS):  # complex spectrum features of the input EEG
    num_classes = segmented_data.shape[0]
    num_chan = segmented_data.shape[1]
    num_trials = segmented_data.shape[2]
    number_of_segments = segmented_data.shape[3]
    fft_len = segmented_data[0, 0, 0, 0, :].shape[0]

    NFFT = round(FFT_PARAMS['sampling_rate'] / FFT_PARAMS['resolution'])
    fft_index_start = int(round(FFT_PARAMS['start_frequency'] / FFT_PARAMS['resolution']))
    fft_index_end = int(round(FFT_PARAMS['end_frequency'] / FFT_PARAMS['resolution'])) + 1

    features_data = np.zeros((2 * (fft_index_end - fft_index_start),
                              segmented_data.shape[1], segmented_data.shape[0],
                              segmented_data.shape[2], segmented_data.shape[3]))

    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                for segment in range(0, number_of_segments):
                    temp_FFT = np.fft.fft(segmented_data[target, channel, trial, segment, :], NFFT) / fft_len
                    real_part = np.real(temp_FFT)
                    imag_part = np.imag(temp_FFT)
                    features_data[:, channel, target, trial, segment] = np.concatenate((
                        real_part[fft_index_start:fft_index_end, ],
                        imag_part[fft_index_start:fft_index_end, ]), axis=0)

    return features_data
