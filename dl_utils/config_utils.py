import importlib
import torch
import numpy as np
import random


def import_module(module_name: str, class_name: str):
    """
    Import a module (class_name) from a module_name (path).

    :param module_name:
        Name of the module from where a class should be imported.
    :param class_name:
        Name of the class to import.
    :return:
        class_: type
            Reference to the imported class.
    """
    module_ = importlib.import_module(module_name)
    class_ = getattr(module_, class_name)
    return class_


def check_config_file(config_file):
    """
    Checks basic structure of a config file.

    :param config_file: dict
        Config file to check.
    :return:
        config_file: dict
            Checked config file.
    """
    assert type(config_file) is dict, 'Config file should be a dictionary'

    file_keys = list(config_file.keys())
    base_message = '[Configurator::dl_utils::check_config_file (AssertionError)]: '
    modules = ['model', 'trainer']

    for mdx, module in enumerate(modules):
        if mdx > 2:
            if module not in file_keys: config_file[module] = None
            continue
        assert module in file_keys, base_message + 'Config file should contain keyword "' + module + '"!'
        assert 'module_name' in config_file[module].keys(), \
            base_message + 'Config file should contain keyword "' + module + '": module_name!'
        assert 'class_name' in config_file[module].keys(), \
            base_message + 'Config file should contain keyword "' + module + '": class_name!'
        assert 'params' in config_file[module].keys(),\
            base_message + 'Config file should contain keyword "' + module + '": params!'

    assert 'data_loader' in config_file['trainer'].keys(), \
        base_message + 'Config file should contain keyword data_loader in trainer!'
    assert 'module_name' in config_file['trainer']['data_loader'].keys(), \
        base_message + 'Please use the keywords [module_name] in the loss dictionary'
    assert 'class_name' in config_file['trainer']['data_loader'].keys(), \
        base_message + 'Please use the keywords [class_name] in the loss dictionary'
    training_params = config_file['trainer']['params']
    assert 'optimizer_params' in training_params.keys(), \
        base_message + 'Please use the keywords [optimizer_params] in the training_params dictionary'
    opt_params = training_params['optimizer_params']
    assert 'lr' in opt_params.keys(), \
        base_message + 'Please use the keywords [lr] in the optimizer_params dictionary'
    assert 'loss' in training_params.keys(), \
        base_message + 'Please use the keywords [loss] in the optimizer_params dictionary'
    assert 'module_name' in training_params['loss'].keys(), \
        base_message + 'Please use the keywords [module_name] in the loss dictionary'
    assert 'class_name' in training_params['loss'].keys(), \
        base_message + 'Please use the keywords [class_name] in the loss dictionary'
    return config_file


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
