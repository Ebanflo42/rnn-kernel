import os
import math
import torch
import numpy as np
import numpy.random as rd

from librosa import load
from os.path import join as opj
from joblib import delayed, Parallel
from python_speech_features import mfcc
from prefetch_generator import background


class_names = ["yes", "no", "up", "down", "left", "right",
               "on", "off", "stop", "go", "silent", "unknown"]


def get_class_id_from_path(example_path):
    """
    :param example_path: path to testing example
    :return: class index based on the above list of class names
    """
    for i, name in enumerate(class_names):
        if name in example_path:
            return i
    return 11


def load_background_noise(gsc_path, sample_rate):
    noise_files = [opj(gsc_path, '_background_noise_', f) for f in os.listdir(
        opj(gsc_path, "_background_noise_")) if f.endswith(".wav")]
    return np.concatenate([load(f, sr=sample_rate)[0] for f in noise_files])


def load_test_and_valid_lists(gsc_path):
    with open(opj(gsc_path, "testing_list.txt")) as t:
        test_list = [line.replace("\n", "") for line in t.readlines()]
    with open(opj(gsc_path, "validation_list.txt")) as t:
        valid_list = [line.replace("\n", "") for line in t.readlines()]
    return test_list, valid_list


def load_wav_file(gsc_path, class_name, sample_rate, test_list, valid_list, example_path=None):
    """
    :param gsc_path: Location of the google speech commands.
    :param sample_rate: Desired sample rate for the wav files.
    :param class_name: Name of the command to load.
    :param test_list: Paths of all test examples.
    :param valid_list: Paths of all validation examples.
    :param example_path: If not None, load the specific wav file specified by this path. Otherwise, load a random file in the given class.
    :return: Numpy array of length sample_rate containing all the waveform amplitudes over the course of a second.
    """

    # exactly 1 second of silence
    if class_name == "silent":
        return np.zeros((sample_rate), dtype=np.float32)
    # choose one of the unknown words
    elif class_name == "unknown":
        directories = [d for d in os.listdir(
            gsc_path) if d not in class_names]
        directories = [d for d in directories if d not in [
            "_background_noise_", "LICENSE", "README.md", "testing_list.txt", "validation_list.txt"] and get_class_id_from_path(d) == 11]
        class_name = directories[rd.randint(
            low=0, high=len(directories))]

    # resample until we have an example outside the test and validation sets
    if example_path is None:

        example_path = opj(gsc_path, class_name)
        all_examples = os.listdir(example_path)
        num_examples = len(all_examples)
        example_file = all_examples[rd.random_integers(
            low=0, high=num_examples-1)]

        while opj(class_name, example_file) not in test_list and opj(class_name, example_file) not in valid_list:
            example_file = all_examples[rd.random_integers(
                low=0, high=num_examples-1)]

        example_path = opj(example_path, example_file)

        audio, _ = load(example_path, sr=sample_rate)

    else:
        audio, _ = load(opj(gsc_path, example_path), sr=sample_rate)

    # cut or paste to make the sample exactly 1 second
    if len(audio) >= sample_rate:
        return audio[:sample_rate]
    elif len(audio) < sample_rate:
        return np.concatenate((audio, np.zeros((sample_rate - len(audio)), dtype=np.float32)))
    else:
        return audio


def preprocess_example(audio: np.ndarray, n_features: int,
                       sample_rate: int, window_size: int,
                       window_stride: int, bkg_noise: np.ndarray, epsilon: float):

    noise_ix = rd.random_integers(0, len(bkg_noise) - 2*sample_rate)
    noise = bkg_noise[noise_ix : sample_rate + noise_ix]
    audio += epsilon*noise

    fingerprint = mfcc(audio, samplerate=sample_rate, winlen=window_size/sample_rate,
                       winstep=window_stride/sample_rate, numcep=n_features, nfilt=2*n_features, nfft=1024)

    return fingerprint


def get_google_speech_train_iterator(batch_size: int, n_features: int,
                                     gsc_path: str, sample_rate: int,
                                     window_size: int, window_stride: int, epsilon: float):
    """
    :param batch_size: batch size
    :param n_features: number of frequency features to decompose words into
    :param gsc_path: base directory of google speech commands
    :param sample_rate: sample rate for loading wav files
    :param window_size: MFCC window size
    :param window_stride: MFCC window stride
    :param epsilon: noise amplitude
    :return: training data iterator which loads example batches on the cpu in the background
    """

    # load background noise and normalize volume
    bkg_noise = load_background_noise(gsc_path, sample_rate)
    bkg_noise /= np.std(bkg_noise)

    # heuristic value
    # I found that the std of the actual sample waveforms is about 0.08
    bkg_noise *= 0.08

    test_list, valid_list = load_test_and_valid_lists(gsc_path)

    @background(max_prefetch=8)
    def train_iter():
        while True:
            class_ids = rd.randint(0, 12, size=batch_size)
            wav_samples = [load_wav_file(
                gsc_path, class_names[i], sample_rate, test_list, valid_list) for i in class_ids]
            fingerprints = Parallel(n_jobs=8, prefer='threads')(delayed(preprocess_example)(
                audio, n_features, sample_rate, window_size, window_stride, bkg_noise, epsilon) for audio in wav_samples)
            in_tensor = torch.tensor(fingerprints, dtype=torch.float32)
            targ_tensor = torch.tensor(class_ids, dtype=torch.int64)
            yield in_tensor, targ_tensor

    return train_iter()


def get_google_speech_test_iterator(batch_size: int, n_features: int,
                                    gsc_path: str, sample_rate: int, window_size: int,
                                    window_stride: int, epsilon: float):
    """
    :param batch_size: batch size
    :param n_features: number of frequency features to decompose words into
    :param gsc_path: base directory of google speech commands
    :param sample_rate: sample rate for loading wav files
    :param window_size: MFCC window size
    :param window_stride: MFCC window stride
    :param epsilon: noise amplitude
    :return: testing data iterator which loads example batches on the cpu in the background with adversarial noise added
    """

    # load background noise and normalize volume
    bkg_noise = load_background_noise(gsc_path, sample_rate)
    bkg_noise /= np.std(bkg_noise)

    # heuristic value
    # I found that the std of the actual sample waveforms is about 0.08
    bkg_noise *= 0.08

    test_list, valid_list = load_test_and_valid_lists(gsc_path)

    # randomly filter out unknown examples
    # otherwise they will offset the testing accuracy
    test_paths = [path for path in test_list if get_class_id_from_path(
        path) < 11 or rd.uniform() < 1/11]

    n_test_samples = len(test_paths)

    @background(max_prefetch=8)
    def test_iter():
        for i in range(0, n_test_samples, batch_size):
            this_batch_size = np.minimum(batch_size, n_test_samples - i)
            paths = test_paths[i: i + this_batch_size]
            class_ids = [get_class_id_from_path(path) for path in paths]
            # switch 1/12th of the test examples to silence so as to not corrupt accuracy
            names = [class_names[i] if rd.uniform() < 11/12 else 'silent' for i in class_ids]
            wav_samples = [load_wav_file(
                gsc_path, n, sample_rate, test_list, valid_list, example_path=path) for n, path in zip(names, paths)]
            fingerprints = Parallel(n_jobs=8, prefer='threads')(delayed(preprocess_example)(
                audio, n_features, sample_rate, window_size, window_stride, bkg_noise, epsilon) for audio in wav_samples)
            in_tensor = torch.tensor(fingerprints, dtype=torch.float32)
            targ_tensor = torch.tensor(class_ids, dtype=torch.int64)
            yield in_tensor, targ_tensor

    return test_iter()


def generate_spirals(n_samples: int, length: int = 100):
    """

    :param n_samples: number of samples to generate
    :param length: length of the time sampling of the spirals
    :return: X, y data samples. X is a torch.Tensor of shape (n_samples, length, 2), y is of shape (n_samples) with 0/1
    labels
    """
    t = torch.linspace(0., 4 * math.pi, length)

    start = torch.rand(n_samples) * 2 * math.pi
    width = torch.rand(n_samples)
    speed = torch.rand(n_samples)
    x_pos = torch.cos(start.unsqueeze(1) + speed.unsqueeze(1)
                      * t.unsqueeze(0)) / (1 + width.unsqueeze(1) * t)

    x_pos[:(n_samples // 2)] *= -1
    y_pos = torch.sin(start.unsqueeze(1) + speed.unsqueeze(1)
                      * t.unsqueeze(0)) / (1 + width.unsqueeze(1) * t)

    X = torch.stack([x_pos, y_pos], dim=2)
    y = torch.zeros(n_samples, dtype=torch.int64)
    y[:(n_samples // 2)] = 1

    perm = torch.randperm(n_samples)

    X = X[perm]
    y = y[perm]

    return X, y


def get_data(length: int, batch_size: int, n_train: int = 1000, n_test: int = 1000):
    """Generate train and test dataloaders for the spirals dataset.

    :param length: length of the time sampling of the spirals
    :param batch_size: batch size
    :param n_train: number of training samples
    :param n_test: number of test samples
    :return: train_dataloader, test_dataloader, output_channels. The train and test dataloader are of type
    torch.utils.dataset.DataLoader, output_channels is the number of classes, equal to 2 for the spirals.
    """

    X_train, y_train = generate_spirals(n_train, length=length)
    X_test, y_test = generate_spirals(n_test, length=length)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size)

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader, 2
