"""
Main.py
- main entry point to start DL experiments

"""
import yaml
import logging
import argparse
import wandb
from datetime import datetime
import sys
sys.path.insert(0, '../RA')
from dl_utils.config_utils import *
import warnings


class Main(object):

    def __init__(self, config_file):
        self.config_file = config_file
        super(Main, self).__init__()

    def setup_experiment(self):
        warnings.filterwarnings(action='ignore')
        logging.info("[Main::setup_experiment]: ################ Starting setup ################")
        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
        self.config_file['trainer']['params']['checkpoint_path'] += date_time
        for idx, dst_name in enumerate(self.config_file['downstream_tasks']):
            self.config_file['downstream_tasks'][dst_name]['checkpoint_path'] += date_time

        # Initialize Configurator
        if self.config_file['configurator'] is None:
            logging.info("[Main::setup_experiment::ERROR]: Configurator module not found, exiting...")
            exit()

        configurator_class = import_module(self.config_file['configurator']['module_name'],
                                           self.config_file['configurator']['class_name'])
        configurator = configurator_class(config_file=self.config_file, log_wandb=True)
        exp_name = configurator.dl_config['experiment']['name']
        method_name = configurator.dl_config['name']
        logging.info("[Main::setup_experiment]: ################ Starting experiment * {} * using method * {} * "
                     "################".format(exp_name, method_name))

        config_dict = dict(
            yaml=config_file,
            params=configurator.dl_config
        )

        wandb.init(project=exp_name, name=method_name, config=config_dict, id=date_time)

        device = 'cuda' if config_file['device'] == 'gpu' else 'cpu'
        checkpoint = dict()
        if configurator.dl_config['experiment']['weights'] is not None:
            checkpoint = torch.load(configurator.dl_config['experiment']['weights'], map_location=torch.device(device))

        if configurator.dl_config['experiment']['task'] == 'train':
            configurator.start_training(checkpoint)
        else:
            configurator.start_evaluations(checkpoint['model_weights'])


def add_args(parser):
    """
    parser: argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--log_level', type=str, default='INFO', metavar='L',
                        help='log level from : ["INFO", "DEBUG", "WARNING", "ERROR"]')

    parser.add_argument('--config_path', type=str, default='projects/dummy_project/config/mnist.yaml', metavar='C',
                        help='path to configuration yaml file')

    return parser


if __name__ == "__main__":
    arg_parser = add_args(argparse.ArgumentParser(description='IML-DL'))
    args = arg_parser.parse_args()
    if args.log_level == 'INFO':
        logging.basicConfig(level=logging.INFO)
    elif args.log_level == 'DEBUG':
        logging.basicConfig(level=logging.DEBUG)
    elif args.log_level == 'WARNING':
        logging.basicConfig(level=logging.WARNING)
    elif args.log_level == 'ERROR':
        logging.basicConfig(level=logging.ERROR)
    config_file = None
    logging.info(
        '------------------------------- DEEP LEARNING FRAMEWORK *IML-COMPAI-DL*  -------------------------------')
    try:
        stream_file = open(args.config_path, 'r')
        config_file = yaml.load(stream_file, Loader=yaml.FullLoader)
        logging.info('[IML-COMPAI-DL::main] Success: Loaded configuration file at: {}'.format(args.config_path))
    except:
        logging.error('[IML-COMPAI-DL::main] ERROR: Invalid configuration file at: {}, exiting...'.format(args.config_path))
        exit()
    Main(config_file).setup_experiment()
