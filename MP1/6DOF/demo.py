import torch
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from absl import app, flags
import numpy as np
from torchvision import transforms
import logging
import time
import sys
from dataset import YCBVDataset
from network import SimpleModel
from utils import test, get_metrics, logger, setup_logging, extract_rotation_translation_matrices

import matplotlib.pyplot as plt

# sys.argv = [sys.argv[0]]

FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 1e-4, 'Learning Rate')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight Deacy for optimizer')
flags.DEFINE_string('output_dir', 'runs/basic/', 'Output Directory')
flags.DEFINE_string('data_dir', 'data/ycbv/v1/', 'Output Directory')
flags.DEFINE_integer('batch_size', 16, 'Batch Size')
flags.DEFINE_integer('seed', 2, 'Random seed')
flags.DEFINE_integer('max_iter', 100000, 'Total Iterations')
flags.DEFINE_integer('val_every', 1000, 'Iterations interval to validate')
flags.DEFINE_integer('save_every', 50000, 'Iterations interval to save model')
flags.DEFINE_integer('preload_images', 1, 
    'Weather to preload train and val images at beginning of training.')
flags.DEFINE_multi_integer('lr_step', [60000, 80000], 'Iterations to reduce learning rate')
flags.DEFINE_boolean('use_6d', False, 'Use 6D rotation representation')
flags.DEFINE_boolean('use_seperate_heads', False, 'Use seperate heads for rotation and translation')
flags.DEFINE_boolean('cropping', False, 'Crop images')

log_every = 20

def geodesic_loss(R_pred, R_gt):
    """
    Computes geodesic loss (angular distance) between predicted and ground-truth rotation matrices.

    Args:
        R_pred (torch.Tensor): Predicted rotation matrices, shape (batch_size, 3, 3).
        R_gt (torch.Tensor): Ground-truth rotation matrices, shape (batch_size, 3, 3).

    Returns:
        torch.Tensor: Scalar geodesic loss.
    """

    # Compute relative rotation matrix
    R_diff = R_gt.permute(0, 2, 1) @ R_pred  # R_gt^T * R_pred

    # Compute trace of R_diff
    trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(dim=-1)

    # Compute the rotation angle (clamped for numerical stability)
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))

    # Return mean loss over the batch
    return theta.mean()


def main(_):
    setup_logging(FLAGS.output_dir)
    torch.set_num_threads(4)
    torch.manual_seed(FLAGS.seed)
    # set_seed(FLAGS.seed)
    
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225]),])
    dataset_train = YCBVDataset(split='train', transform=transform,
                                data_dir=FLAGS.data_dir, 
                                preload_images=FLAGS.preload_images,
                                cropping=FLAGS.cropping)
    dataset_val = YCBVDataset(split='val', transform=transform, 
                              data_dir=FLAGS.data_dir, 
                              preload_images=FLAGS.preload_images,
                              cropping=FLAGS.cropping)
    dataset_test = YCBVDataset(split='test', transform=transform,
                               data_dir=FLAGS.data_dir,
                               preload_images=FLAGS.preload_images,
                               cropping=FLAGS.cropping)
    dataloader_train = DataLoader(dataset_train, batch_size=FLAGS.batch_size,
                                  num_workers=2, shuffle=True, drop_last=True)
    
    num_classes = dataset_train.num_classes
    device = torch.device('cuda:0')
    model = SimpleModel(num_classes=num_classes, use_6d=FLAGS.use_6d, use_seperate_heads=FLAGS.use_seperate_heads)
    model.to(device)

    writer = SummaryWriter(FLAGS.output_dir, max_queue=1000, flush_secs=120)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, 
                                  weight_decay=FLAGS.weight_decay)
    
    milestones = [int(x) for x in FLAGS.lr_step]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1)
    
    previous_best_val_metric = -1
    
    optimizer.zero_grad()
    dataloader_iter = None
    
    times_np, cls_loss_np, R_loss_np, t_loss_np, total_loss_np = [], [], [], [], []
    metrics_np = {
        'cls_accuracy': [],
        'cls_R_accuracy': [],
        'cls_t_accuracy': [],
        'cls_R_t_accuracy': [],
        'overall': [],
    }
     
    for i in range(FLAGS.max_iter):
        iter_start_time = time.time()
        
        if dataloader_iter is None or i % len(dataloader_iter) == 0:
            dataloader_iter = iter(dataloader_train)
        image, bbox, cls_gt, R_gt, t_gt, key_name = next(dataloader_iter) 

        # image_to_show = image[2].permute(1, 2, 0).cpu().numpy()
        
        # if image_to_show.min() < 0:
        #     image_to_show = (image_to_show - image_to_show.min()) / (image_to_show.max() - image_to_show.min())

        # # Show image
        # plt.imshow(image_to_show)
        # plt.title(f'Input Image {i}')
        # plt.axis("off")
        # plt.show()

        image = image.to(device, non_blocking=True)
        
        bbox = bbox.to(device, non_blocking=True)
        cls_gt = cls_gt.to(device, non_blocking=True)
        R_gt = R_gt.to(device, non_blocking=True)
        t_gt = t_gt.to(device, non_blocking=True)

        logits, R, t = model(image, bbox)
        
        # Compute metrics
        cls_pred, R_pred, t_pred = model.process_output((logits, R, t))
        # if FLAGS.use_seperate_heads:
        #     R, t = extract_rotation_translation_matrices(cls_pred, R, t, 9, 3)
        metrics = get_metrics(
            cls=cls_pred, R=R_pred, t=t_pred, 
            gt_cls=cls_gt, gt_R=R_gt, gt_t=t_gt)        
        for key, value in metrics.items():
            metrics_np[key].append(value)

        # Loss functions for training
        classification_loss = nn.CrossEntropyLoss()(logits, cls_gt)

        if FLAGS.use_seperate_heads:
            logic_mask = cls_pred == cls_gt
            R = R[logic_mask]
            t = t[logic_mask]
            R_gt = R_gt[logic_mask]
            t_gt = t_gt[logic_mask]

        if FLAGS.use_6d:
            R_loss = geodesic_loss(R.reshape(-1, 3, 3), R_gt.reshape(-1, 3, 3))
        else:
            R_loss = nn.MSELoss()(R, R_gt.reshape(-1, 9))
        t_loss = nn.MSELoss()(t, t_gt.reshape(-1, 3))
        # logging.info(f'classification_loss_shape: {classification_loss.shape}, R_loss_shape: {R_loss.shape}, t_loss_shape: {t_loss.shape}')
        classification_loss = classification_loss.mean()
        R_loss = R_loss.mean()
        t_loss = t_loss.mean()

        total_loss = classification_loss
        if not torch.isnan(R_loss):
          total_loss += R_loss
        if not torch.isnan(t_loss):
          total_loss += t_loss
        
                
        if np.isnan(total_loss.item()):
            logging.error(f'Loss went to NaN at iteration {i+1}')
            break
        
        if np.isinf(total_loss.item()):
            logging.error(f'Loss went to Inf at iteration {i+1}')
            break
        
        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Some logging
        lr = scheduler.get_last_lr()[0]
        total_loss_np.append(total_loss.item())
        cls_loss_np.append(classification_loss.item())
        R_loss_np.append(R_loss.item())
        t_loss_np.append(t_loss.item())
        times_np.append(time.time() - iter_start_time)
                      
        if (i+1) % log_every == 0:
            print('')
            writer.add_scalar('iteration_rate', len(times_np) / np.sum(times_np), i+1)
            logger('iteration_rate', len(times_np) / np.sum(times_np), i+1)
            writer.add_scalar('loss/R', np.mean(R_loss_np), i+1)
            logger('loss/R', np.mean(R_loss_np), i+1)
            writer.add_scalar('loss/t', np.mean(t_loss_np), i+1)
            logger('loss/t', np.mean(t_loss_np), i+1)
            writer.add_scalar('lr', lr, i+1)
            logger('lr', lr, i+1)
            writer.add_scalar('loss/cls', np.mean(cls_loss_np), i+1)
            logger('loss/cls', np.mean(cls_loss_np), i+1)
            writer.add_scalar('loss/total', np.mean(total_loss_np), i+1)
            logger('loss/total', np.mean(total_loss_np), i+1)

            for key, value in metrics_np.items():
                writer.add_scalar(f'metrics/{key}', np.mean(value), i+1)
                logger(f'metrics/{key}', np.mean(value), i+1)

            times_np, cls_loss_np, R_loss_np, t_loss_np, total_loss_np = [], [], [], [], []
            metrics_np = {
                'cls_accuracy': [], 'cls_R_accuracy': [], 'cls_t_accuracy': [],
                'cls_R_t_accuracy': [], 'overall': [],
            }

        if (i+1) % FLAGS.save_every == 0:
            torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_{i+1}.pth')
            
        if (i+1) % FLAGS.val_every == 0 or (i+1) == FLAGS.max_iter:
            print('')
            logging.info(f'Validating at {i+1} iterations.')
            val_dataloader = DataLoader(dataset_val, batch_size=1, num_workers=0)
            result_file_name = f'{FLAGS.output_dir}/predictions_{i+1:06d}_val.json'
            model.eval()

            results, metrics_np_val = test(val_dataloader, device, model, 
                     result_file_name)
            for key, value in metrics_np_val.items():
                writer.add_scalar(f'metrics/val-{key}', np.mean(value), i+1)
                logger(f'metrics/val-{key}', np.mean(value), i+1)
            val_metric = np.mean(metrics_np_val['overall'])
            
            if val_metric > previous_best_val_metric:
                print('')
                logging.info(f'Best validation metric improved from {previous_best_val_metric} to {val_metric}. Saving predictions on test set.')
                test_dataloader = DataLoader(dataset_test, num_workers=0,
                                 shuffle=False, drop_last=False)
                result_file_name = f'{FLAGS.output_dir}/predictions_{i+1:06d}_test.json'
                model.eval()
                test(test_dataloader, device, model, result_file_name)
                previous_best_val_metric = val_metric
                
            model.train()

    torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_final.pth')

    # Save prediction result on test set
    test_dataloader = DataLoader(dataset_test, num_workers=0,
                                 shuffle=False, drop_last=False)
    result_file_name = f'{FLAGS.output_dir}/predictions_{i+1:06d}_test.json'
    model.eval()
    test(test_dataloader, device, model, result_file_name)

if __name__ == '__main__':
    app.run(main)
