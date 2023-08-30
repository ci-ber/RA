"""
DataLoader.py (pl.LightningDataModule)

Default class for data loaders from:
- csv file
- files in directory
"""
from transforms.preprocessing import AddChannelIfNeeded, AssertChannelFirst, ReadImage, To01
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from dl_utils.data_utils import *
from dl_utils.config_utils import *
import logging
import glob
import copy


class DefaultDataset(Dataset):

    def __init__(self, data_dir, file_type='', label_dir=None, target_size=(64, 64), test=False):
        """
        @param data_dir: str
            path to directory or csv file containing data
        @param: file_type: str
            ending of the files, e.g., '*.jpg'
        @param: label_dir: str
            path to directory or csv file containing labels
        @param: image_transform: transform function, default: None
            image transforms, e.g., loading, resize, etc...
        @param: label_transform: transform function, default: None
            label_transform, e.g., loading, resize, etc...
        @param: target_size: tuple (int, int), default: (64, 64)
            the desired output size
        """
        # print('*** Default Dataset##################################')
        # print(data_dir, test)
        super(DefaultDataset, self).__init__()
        self.label_dir = label_dir
        self.target_size = target_size
        if 'csv' in data_dir[0]:
            self.files = get_data_from_csv(data_dir)
        else:
            self.files = [glob.glob(data_dir_i + file_type) for data_dir_i in data_dir]
        self.nr_items = len(self.files)
        logging.info('DefaultDataset::init(): Loading {} files from: {}'.format(self.nr_items, data_dir))

        self.im_t = self.get_image_transform_test() if test else self.get_image_transform()
        # print(data_dir, test)
        # print(self.im_t)
        if label_dir is not None:
            if 'csv' in label_dir[0]:
                self.label_files = get_data_from_csv(label_dir)
            else:
                self.label_files = [glob.glob(label_dir_i + file_type) for label_dir_i in label_dir]
            self.seg_t = self.get_label_transform_test() if test else self.get_label_transform()

    def get_image_transform(self):
        """
        Add specific annotations for training, e.g., augmentation
        """
        default_t_train = transforms.Compose([ReadImage(), To01(), AddChannelIfNeeded(),
                                        AssertChannelFirst(), transforms.Resize(self.target_size)])
        return default_t_train

    def get_image_transform_test(self):
        default_t_test = transforms.Compose([ReadImage(), To01(), AddChannelIfNeeded(),
                                        AssertChannelFirst(), transforms.Resize(self.target_size)])
        return default_t_test

    def get_label_transform(self):
        default_lt = transforms.Compose([ReadImage(), To01(), AddChannelIfNeeded(),
                                        AssertChannelFirst(), transforms.Resize(self.target_size)])
        return default_lt

    def get_label_transform_test(self):
        default_lt_test = transforms.Compose([ReadImage(), To01(), AddChannelIfNeeded(),
                                        AssertChannelFirst(), transforms.Resize(self.target_size)])
        return default_lt_test

    def get_label(self, idx):
        if self.label_dir is not None:
            return self.seg_t(self.label_files[idx])
        else:
            return 0

    def __getitem__(self, idx):
        return self.im_t(self.files[idx]), self.get_label(idx)

    def __len__(self):
        return self.nr_items


class DefaultDataLoader(pl.LightningDataModule):

    def __init__(self, args):
        super(DefaultDataLoader, self).__init__()
        akeys = args.keys()
        dataset_module = args['dataset_module'] if 'dataset_module' in akeys else None
        self.data_dir = args['data_dir'] if 'data_dir' in akeys else None
        self.file_type = args['file_type'] if 'file_type' in akeys else ''
        self.label_dir = args['label_dir'] if 'label_dir' in akeys else {'train': None, 'val': None, 'test': None}
        self.mask_dir = args['mask_dir'] if 'mask_dir' in akeys else {'train': None, 'val': None, 'test': None}
        self.target_size = args['target_size'] if 'target_size' in akeys else (64, 64)
        self.batch_size = args['batch_size'] if 'batch_size' in akeys else 8
        self.num_workers = args['num_workers'] if 'num_workers' in akeys else 2
        assert type(self.data_dir) is dict, 'DefaultDataset::init():  data_dir variable should be a dictionary'
        if dataset_module is not None:
            assert 'module_name' in dataset_module.keys() and 'class_name' in dataset_module.keys(),\
                'DefaultDataset::init(): Please use the keywords [module_name|class_name] in the dataset_module dictionary'
            self.ds_module = import_module(dataset_module['module_name'], dataset_module['class_name'])
            print(dataset_module['class_name'])
        else:
            self.ds_module = import_module('core.DataLoader', 'DefaultDataset')
            print('Default DATASET!')


    def train_dataloader(self):
        if 'train' not in self.data_dir.keys():
            return self.test_dataloader()
        ds_mod = copy.deepcopy(self.ds_module)
        return DataLoader(ds_mod(self.data_dir['train'], self.file_type, self.label_dir['train'],
                                         self.mask_dir['train'], self.target_size, test=False),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True,
                          )

    def val_dataloader(self):
        if 'val' not in self.data_dir.keys():
            return self.test_dataloader()
        ds_mod_val = copy.deepcopy(self.ds_module)
        return DataLoader(ds_mod_val(self.data_dir['val'], self.file_type, self.label_dir['val'],
                                         self.mask_dir['val'], self.target_size, test=True),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True,
                          )

    def test_dataloader(self):
        assert 'test' in self.data_dir.keys(), \
            'DefaultDatasets::init():  Please use the keywords [test] in the data_dir dictionary'
        ds_mod_test = copy.deepcopy(self.ds_module)
        return DataLoader(ds_mod_test(self.data_dir['test'], self.file_type, self.label_dir['test'],
                                         self.mask_dir['test'], self.target_size, test=True),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True,
                          )
