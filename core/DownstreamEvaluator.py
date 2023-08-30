"""
DownstreamEvaluator.py

Run Downstream Tasks after training has finished
"""
import os


class DownstreamEvaluator(object):
    """
    Downstream Tasks
        - run tasks at training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """

    def __init__(self, name, model, device, test_data_dict, checkpoint_path):
        """
        @param model: nn.Module
            the neural network module
        @param device: torch.device
            cuda or cpu
        @param test_data_dict:  dict(datasetname, datasetloader)
            dictionary with dataset names and loaders
        @param checkpoint_path: str
            path to save results
        """
        self.name = name
        self.model = model.to(device)
        self.device = device
        self.test_data_dict = test_data_dict
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.image_path = checkpoint_path + '/images/'
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)
        super(DownstreamEvaluator, self).__init__()

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
            dictionary with the model weights of the federated collaborators
        """
        raise NotImplementedError("[DownstreamEvaluator::start_task]: Please Implement start_task() method")