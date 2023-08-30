"""
Configurator.py

Default class for configuring DL experiments

"""
import copy
import logging
import torch

from dl_utils.config_utils import check_config_file, import_module, set_seed


class DLConfigurator(object):
    """
    DL Configurator
        - parametrization of deep learning via 'config.yaml'
        - initializes Trainer:
            routine for training neural networks
        - initializes FedDownstreamTask:
            routing for downstream evaluations, e.g., classification, anomaly detection
        - starts the experiments
    """
    def __init__(self, config_file, log_wandb=False):
        """
        :param config_file: file
            config.yaml file that contains the DL configuration
        """
        self.dl_config = check_config_file(config_file)
        if self.dl_config is None:
            print('[Configurator::init] ERROR: Invalid configuration file. Configurator will exit...')
            return

        # set seeds
        set_seed(2109)

        # init model and device
        dev = self.dl_config['device']
        self.device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu') if dev == 'gpu' else 'cpu'
        model_class = import_module(self.dl_config['model']['module_name'], self.dl_config['model']['class_name'])
        self.model = model_class(**(self.dl_config['model']['params']))

    def start_training(self, global_model: dict = dict()):
        # Trainer Init
        data = self.load_data(self.dl_config['trainer']['data_loader'], train=True)
        trainer_class = import_module(self.dl_config['trainer']['module_name'], self.dl_config['trainer']['class_name'])
        self.trainer = trainer_class(training_params=self.dl_config['trainer']['params'], model=copy.deepcopy(self.model),
                                     data=data, device=self.device, log_wandb=True)
        # Train
        model_state, opt_state, epoch = None, None, 0
        if self.trainer is None:
            logging.error("[Configurator::train::ERROR]: Trainer not defined! Shutting down...")
            exit()
        if 'model_weights' in global_model.keys():
            model_state = global_model['model_weights']
            logging.info("[Configurator::train::INFO]: Model weights loaded!")
        if 'optimizer_weights' in global_model.keys():
            opt_state = global_model['optimizer_weights']
        if 'epoch' in global_model.keys():
            epoch = global_model['epoch']

        logging.info("[Configurator::train]: ################ Starting training ################")
        trained_model_state, trained_opt_state = self.trainer.train(model_state, opt_state, epoch)
        logging.info("[Configurator::train]: ################ Finished training ################")
        logging.info("[Configurator::train]: ################ Starting testing ################")
        self.trainer.test(trained_model_state, data.test_dataloader(), task='Test')
        logging.info("[Configurator::train]: ################ Finished testing ################")
        self.start_evaluations(trained_model_state)

    def start_evaluations(self, global_model):
        # Downstream Tasks
        self.downstream_tasks = []
        nr_tasks = len(self.dl_config['downstream_tasks'])
        for idx, dst_name in enumerate(self.dl_config['downstream_tasks']):
            logging.info("[Configurator::eval]: ################ Starting downstream task nr. {}/{} -- {}-- ################"
                         .format(idx+1, nr_tasks, dst_name))
            dst_config = self.dl_config['downstream_tasks'][dst_name]
            downstream_class = import_module(dst_config['module_name'], dst_config['class_name'])
            data = self.load_data(dst_config['data_loader'], train=False)

            if 'params' in dst_config.keys():
                dst = downstream_class(dst_name, self.model, self.device, data, dst_config['checkpoint_path'],
                                       **dst_config['params'])
            else:
                dst = downstream_class(dst_name, self.model, self.device, data, dst_config['checkpoint_path'])
            dst.start_task(global_model=global_model)
            logging.info("[Configurator::eval]: ################ Finished downstream task nr. {}/{} ################"
                         .format(idx, nr_tasks))

    @staticmethod
    def load_data(data_loader_config, train=True):
        """
        :param data_loader_config: dict
            parameters for data loaders -  must include module/class name and params
        :param train: bool
            True if the datasets are used for training, False otherwise
        :return: list of
           train, val, test datasets if train is True, dict with downstream datasets otherwise
        """
        data_loader_module = import_module(data_loader_config['module_name'], data_loader_config['class_name'])

        if train:
            return data_loader_module(**(data_loader_config['params']))

        downstream_datasets = dict()
        for dataset_name in data_loader_config['datasets']:
            data = data_loader_module({**(data_loader_config['params']['args']), **(data_loader_config['datasets'][dataset_name])})
            downstream_datasets[dataset_name] = data.test_dataloader()
        return downstream_datasets



