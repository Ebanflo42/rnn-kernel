import os
from typing import List

import numpy as np
import pandas as pd
import sacred
import torch

import adversarial
import generate_data
import rnn
import utils


ex = sacred.Experiment()


# Configuration
@ex.config
def my_config():
    non_linearity = 'tanh'
    batch_size = 64
    reg_lambda = 0.
    hidden_channels = 800
    n_epoch = 10
    order = 0
    length = 100
    n_train = 100
    n_test = 100
    lr = None
    save_dir = ''
    adversarial_epsilon = [0.]
    adversarial_steps = 10

    # how many training batches to show the network
    n_steps = 20

    # path to google speech commands
    gsc_path = '/home/medusa/Data/speech_commands_v0.02'

    # number of frequency features to be extracted from speech commands
    n_features = 40

    # samples per second of wav file (each command is one second long)
    sample_rate = 16000

    # window size and stride in samples for frequency extraction
    window_size = 500
    window_stride = 20

    # some amount of noise perturbation is needed during training to prevent overfitting
    base_epsilon = 0.0

    save_every = 1


@ex.main
def train_model(_run: sacred.Experiment, non_linearity: str, batch_size: int, reg_lambda: float, hidden_channels: int,
                n_epoch: int, save_every: int, order: int, length: int, n_train: int, n_test: int, lr: float, save_dir: str, n_steps: int,
                gsc_path: str, n_features: int, sample_rate: int, window_size: int, window_stride: int, base_epsilon: float) \
        -> torch.nn.Module:
    """Main function of the adversarial experiment that generates spirals and train a RNN, with or without penalization.
    The results of the run are stored in the files config.json and metrics.json.

    :param _run: run ID
    :param non_linearity: activation function of the RNN must be 'tanh' or 'sigmoid'
    :param batch_size: batch size
    :param reg_lambda: regularization parameter, can be None or float. If None, no regularization is applied.
    :param hidden_channels: size of the hidden state of the RNN
    :param n_epoch: number of training epochs
    :param order: truncation order for computing the norm in the RKHS as a N-step Taylor expansion
    :param length: number of sampling points of the spirals
    :param n_train: number of training samples
    :param n_test: number of test samples
    :param lr: learning rate
    :param save_dir: path to directory where the experiment is saved
    :return: Trained RNN.
    """
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)
    ex_save_dir = os.path.join(save_dir, _run._id)

    # train_dataloader, test_dataloader, output_channels = generate_data.get_data(
    #    length, batch_size, n_train=n_train, n_test=n_test)
    train_dataloader = generate_data.get_google_speech_train_iterator(
        batch_size, n_features, gsc_path, sample_rate, window_size, window_stride, base_epsilon)
    test_dataloader = generate_data.get_google_speech_test_iterator(
        batch_size, n_features, gsc_path, sample_rate, window_size, window_stride, 0)
    output_channels = 12

    input_channels = next(iter(train_dataloader))[
        0].shape[2]  # get last dimension of one batch
    _run.log_scalar('input_channels', input_channels)
    _run.log_scalar('output_channels', output_channels)

    model = rnn.RNNModel(input_channels, hidden_channels,
                         output_channels, non_linearity=non_linearity, device=device)
    model.to(device)

    rnn.train_penalized_rnn(model, train_dataloader, n_steps, save_every=save_every, n_epoch=n_epoch, verbose=True, reg_lambda=reg_lambda, order=order,
                            save_dir=ex_save_dir, device=device, lr=lr)

    test_acc = utils.evaluate_model(model, test_dataloader, steps=10, device=device)
    train_acc = utils.evaluate_model(model, train_dataloader, steps=10, device=device)

    _run.log_scalar('accuracy_test', test_acc.item())
    _run.log_scalar('accuracy_train', train_acc.item())

    return model


@ex.capture
def compute_adversarial_accuracy(experiment_dir: str, batch_size: int,
                                 gsc_path: str, n_features: int, sample_rate: int, window_size: int, window_stride: int, n_steps: int = None, run_nums: List[str] = None):
    """Generates adversarial test examples for several runs in an experiment and save the resulting accuracy in a
    dataframe.

    :param experiment_dir: path to the directory where the experiment is saved
    :param run_nums: run ids to consider. If None all the runs in the experiment_dir are tested.
    :return: None
    """
    df = utils.get_ex_results(experiment_dir, run_nums)

    df_adv = pd.DataFrame(columns=['acc_test_adv', 'epsilon', 'reg_lambda'])
    n_runs = df.shape[0]
    for index, exp in df.iterrows():
        print('Computing adversarial accuracy on experiment {}/{}'.format(index, n_runs))
        for epsilon in exp['adversarial_epsilon']:
            print('epsilon: {}'.format(epsilon))
            model = utils.get_RNN_model(exp, step=n_steps)
            """
            X_test, y_test = generate_data.generate_spirals(
                exp['n_test'], length=exp['length'])
            if epsilon > 0.:
                pgd_1 = adversarial.PGDL2(
                    model, epsilon, steps=exp['adversarial_steps'])
                X_test = pgd_1(X_test, y_test)
            test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset, batch_size=exp['batch_size'])
            """
            test_dataloader = generate_data.get_google_speech_test_iterator(
                batch_size, n_features, gsc_path, sample_rate, window_size, window_stride, epsilon)
            acc_test = utils.evaluate_model(model, test_dataloader, steps=1)
            df_adv = df_adv.append({'epsilon': epsilon, 'acc_test_adv': float(acc_test),
                                    'reg_lambda': exp['reg_lambda']}, ignore_index=True)

    df_adv.loc[df_adv['reg_lambda'] == 0., 'model'] = 'RNN'
    df_adv.loc[df_adv['reg_lambda'] != 0., 'model'] = 'Penalized RNN'
    df_adv.to_csv(os.path.join(experiment_dir, 'adversarial_accuracy.csv'))


@ex.capture
def compute_norms(experiment_dir: str, run_nums: List[str] = None):
    """"Computes the RKHS norm and the Frobenius norm of the weights during training of the runs in run_nums.

    :param experiment_dir: path to the directory where the experiment is saved
    :param run_nums: run ids to consider. If None all the runs in the experiment_dir are tested.
    :return: None
    """
    df = utils.get_ex_results(experiment_dir, run_nums)

    df_norms = pd.DataFrame(
        columns=['norm_kernel', 'norm_frobenius', 'reg_lambda', 'epoch'])
    n_runs = df.shape[0]
    for index, exp in df.iterrows():
        print('Computing norms on experiment {}/{}'.format(index, n_runs))
        n_epoch = exp['n_epoch']
        norm_kernel = []
        norm_frobenius = []
        for i in range(n_epoch):
            model = utils.get_RNN_model(exp, i)
            norm_kernel += [float(torch.norm(model.get_kernel_penalization(3)))]
            norm_frobenius += [
                float(torch.norm(torch.cat([model.weight_ih, model.weight_hh], 1)))]
        norm_kernel_smoothed = np.convolve(
            norm_kernel, np.ones(6)/6, mode='same')
        df_norms = df_norms.append(pd.DataFrame({'norm_kernel': norm_kernel, 'norm_frobenius': norm_frobenius,
                                                 'norm_kernel_smoothed': norm_kernel_smoothed,
                                                 'epoch': np.arange(n_epoch),
                                                 'reg_lambda': np.full(n_epoch, exp['reg_lambda'])}),
                                   ignore_index=True)
    df_norms.loc[df_norms['reg_lambda'] == 0., 'model'] = 'RNN'
    df_norms.loc[df_norms['reg_lambda'] != 0., 'model'] = 'Penalized RNN'
    df_norms.to_csv(os.path.join(experiment_dir, 'training_norms.csv'))
