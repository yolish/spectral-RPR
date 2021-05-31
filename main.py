"""
Cloned from: https://github.com/yolish/transposenet and modified for testing smooth camera pose regression
Entry point training and testing
"""

import argparse
import torch
import numpy as np
import json
import logging
from util import utils
from models.SpectralRPR import SpectralRPR
from train import train
from test import test

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_name",
                            help="name of model to create (e.g. posenet, transposenet")
    arg_parser.add_argument("mode", help="train or eval")
    arg_parser.add_argument("backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("train_labels_file", help="path to a file mapping images to their poses for the train dataset")
    arg_parser.add_argument("--rpr_train_dataset_embedding_path",
                            help="path to a file with the embedding of the train dataset embedding for RPR")
    arg_parser.add_argument("--ir_train_dataset_embedding_path",
                            help="path to a file with the embedding of the train dataset embedding for IR")
    arg_parser.add_argument("--test_labels_file",
                            help="path to a file mapping images to their poses for the test dataset")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {} with {}".format(args.model_name, args.mode))
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    logging.info("Using dataset: {}".format(args.dataset_path))

    # Read configuration
    with open('config.json', "r") as read_file:
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
    model = SpectralRPR(config).to(device)
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    if args.mode == 'train':
        train(model, config, device, args.dataset_path, args.train_labels_file, args.ir_train_dataset_embedding_path)
    else: # Test
        stats = test(model, config, device, args.dataset_path, args.train_labels_file, args.test_labels_file,
                     args.rpr_train_dataset_embedding_path, args.ir_train_dataset_embedding_path)
        # Record overall statistics
        logging.info("Performance of {} on {}".format(args.checkpoint_path, args.test_labels_file))
        logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.median(stats[:, 0]), np.median(stats[:, 1])))
        logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))






