import torch
from datasets.KRPRDataset import KRPRDataset
from util import utils
from os.path import join
from models.pose_losses import ExtendedCameraPoseLoss
import logging


def train(model, config, device, dataset_path, labels_file, embedding_path):
    """
    model: (nn.Module) RPR spectral model
    config: config object for set up
    device: (torch.device)
    dataset_path: (str) path to train dataset (where images are located)
    labels_file (str) path to a file associating images with their absolute poses
    embedding_path (str) path to embedding of the dataset for knn retrieval
    """

    # Set to train mode
    model.train()
    pose_loss = ExtendedCameraPoseLoss(config).to(device)

    # Set the optimizer and scheduler
    params = list(model.parameters()) + list(pose_loss.parameters())
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                              lr=config.get('lr'),
                              eps=config.get('eps'),
                              weight_decay=config.get('weight_decay'))
    scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                step_size=config.get('lr_scheduler_step_size'),
                                                gamma=config.get('lr_scheduler_gamma'))

    # Set the dataset and data loader
    k = config.get("k")
    transform = utils.train_transforms.get('baseline')
    dataset = KRPRDataset(dataset_path, labels_file, embedding_path, k, transform)
    loader_params = {'batch_size': config.get('batch_size'),
                              'shuffle': True,
                              'num_workers': config.get('n_workers')}
    dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

    # Get training details
    n_freq_print = config.get("n_freq_print")
    n_freq_checkpoint = config.get("n_freq_checkpoint")
    n_epochs = config.get("n_epochs")

    # Train
    checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
    n_total_samples = 0.0
    loss_vals = []
    sample_count = []
    for epoch in range(n_epochs):

        # Resetting temporal loss used for logging
        running_loss = 0.0
        n_samples = 0

        for batch_idx, minibatch in enumerate(dataloader):
            for key, value in minibatch.items():
                minibatch[key] = value.to(device)
            gt_abs_poses = minibatch.get('pose').to(dtype=torch.float32)
            batch_size = gt_abs_poses.shape[0]
            gt_rel_poses = minibatch.get('rel_query_knn_poses').to(dtype=torch.float32)
            gt_rel_poses = gt_rel_poses.view(batch_size*k, gt_rel_poses.shape[2])

            n_samples += batch_size
            n_total_samples += batch_size

            # Zero the gradients
            optim.zero_grad()

            # Forward pass
            res = model(minibatch)
            est_abs_poses = res.get('abs_poses')
            est_rel_poses = res.get('rel_poses')

            criterion = pose_loss(est_abs_poses, gt_abs_poses, est_rel_poses, gt_rel_poses)

            # Collect for recoding and plotting
            running_loss += criterion.item()
            loss_vals.append(criterion.item())
            sample_count.append(n_total_samples)

            # Back prop
            criterion.backward()
            optim.step()

            # Record loss and performance on train set
            if batch_idx % n_freq_print == 0:
                posit_err, orient_err = utils.pose_err(est_abs_poses.detach(), gt_abs_poses.detach())

                logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                             "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                    batch_idx+1, epoch+1, (running_loss/n_samples),
                                                                    posit_err.mean().item(),
                                                                    orient_err.mean().item()))
        # Save checkpoint
        if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
            torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

        # Scheduler update
        scheduler.step()

    logging.info('Training completed')
    torch.save(model.state_dict(), checkpoint_prefix + '_final.pth'.format(epoch))

    loss_fig_path = checkpoint_prefix + "_loss_fig.png"
    utils.plot_loss_func(sample_count, loss_vals, loss_fig_path)