import logging
import os
from pathlib import Path
import pickle

import click
import torch

import adversarial_experiment
import configurations
import plots
import taylor_experiment
import utils


@click.command()
@click.option('--taylor-exp', type=str, default=None, help='Should run the Taylor expansion experiment.')
@click.option('--adversarial-exp', type=str, default=None, help='Should run the adversarial accuracy experiment.')
def main(taylor_exp: str, adversarial_exp: str):
    """ Main function that runs either the taylor or the adversarial experiment.

    :param taylor_exp: str, type of taylor experiment to run. Must be 'test' or 'final'
    :param adversarial_exp: str, type of adversarial experiment to run. Must be 'test' or 'spirals_penalization'
    :return:None
    """
    ROOT_DIR = str(Path(__file__).resolve().parents[0])

    if taylor_exp is not None:
        print('Running the Taylor expansion experiment')
        if taylor_exp not in configurations.configs_taylor_exp:
            raise ValueError('--taylor-exp should be among {}'.format(list(configurations.configs_taylor_exp.keys())))
        config = configurations.configs_taylor_exp[taylor_exp]
        save_dir = os.path.join(ROOT_DIR, 'results', 'taylor', taylor_exp)
        utils.clean_dir(save_dir)
        pickle.dump(config, open(os.path.join(save_dir, 'config.pkl'), 'wb'))

        print('Computing Taylor convergence')
        taylor_experiment.compute_taylor_convergence(save_dir, config)

        print('Plotting results')
        logging.disable(logging.WARNING)  # Disable matplotlib warnings
        plots.plot_taylor_convergence(save_dir)
        logging.disable(logging.NOTSET)  # Re-enable warnings

    if adversarial_exp is not None:
        print('Running the adversarial accuracy experiment')
        if adversarial_exp not in configurations.configs_adversarial_exp:
            raise ValueError('--adversarial-exp should be among {}'.format(list(configurations.configs_adversarial_exp.keys())))
        config = configurations.configs_adversarial_exp[adversarial_exp]
        save_dir = os.path.join(ROOT_DIR, 'results', 'adversarial', adversarial_exp)
        utils.clean_dir(save_dir)
        config['save_dir'] = [save_dir]

        print('Training model')
        utils.gridsearch(adversarial_experiment.ex, config, save_dir)

        print('Computing adversarial accuracy')
        adversarial_experiment.compute_adversarial_accuracy(save_dir, n_steps=config['n_steps'][0])

        print('Computing training norms')
        adversarial_experiment.compute_norms(save_dir)

        print('Plotting results')
        logging.disable(logging.WARNING)  # Disable matplotlib warnings
        plots.plot_spirals_adversarial(save_dir)
        plots.plot_spirals_training_norms(save_dir)
        logging.disable(logging.NOTSET)  # Re-enable warnings


if __name__ == '__main__':
    # Check if GPU is used
    print('GPU available: ', torch.cuda.is_available())
    main()
