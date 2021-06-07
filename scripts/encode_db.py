"""
Encode a dataset
"""

import argparse
import torch
import numpy as np
import json
import logging
from util import utils
from models.SpectralRPR import SpectralRPR
from datasets.CameraPoseDataset import CameraPoseDataset

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_name",
                            help="name of model to create (e.g. posenet, transposenet")
    arg_parser.add_argument("backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("train_labels_file", help="same file used for training (same order)")
    arg_parser.add_argument("checkpoint_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("output_embedding_path",
                            help="path to a .pth file with the embedding of the train dataset will be saved")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Using {} to encode a dataset of images".format(args.model_name))
    logging.info("Using dataset: {}".format(args.dataset_path))

    # Read configuration
    with open('../config.json', "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.model_name]
    general_params = config['general']
    config = {**model_params, **general_params}
    config["backbone"] = args.backbone_path
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Create the model
    #TODO support other model creation
    model = SpectralRPR(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
    logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    # Set to eval mode
    model.eval()

    # Set the dataset and data loader
    transform = utils.test_transforms.get('baseline')
    dataset = CameraPoseDataset(args.dataset_path, args.train_labels_file, transform)
    loader_params = {'batch_size': 1,
                     'shuffle': False,
                     'num_workers': config.get('n_workers')}
    dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

    n = len(dataset)
    db_encoding = torch.zeros((n, model.get_encoding_dim())).to(device)
    with torch.no_grad():
        for i, minibatch in enumerate(dataloader, 0):
            for key, value in minibatch.items():
                minibatch[key] = value.to(device)

            # Embed the image
            logging.info("Encoding image {}/{} at path: {}".format(i+1,n,dataset.img_paths[i]))
            db_encoding[i,:] = model.forward_backbone(minibatch.get('img')).squeeze(0)
            logging.info("Encoding image completed")

    logging.info("Saving encoding to file: {}".format( args.output_embedding_path))
    torch.save(db_encoding, args.output_embedding_path)





