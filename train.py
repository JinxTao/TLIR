import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
# from tensorboardX import SummaryWriter
import os
import numpy as np
from model import TLIR
import loss.dpssim as dpss
import torch.nn as nn 




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/TLIR_256_256_90.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))


    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase,'train')
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase,'valid')
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = TLIR.TLIR(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    optim_params = []
    optim_params.append(diffusion.finaModel.parameters())
    optim_params.append(diffusion.namda)

    optimizer = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])

    loss_func = nn.L1Loss(reduction='sum').to(diffusion.device)
    dpss_func = dpss.CombinedLoss(device=diffusion.device).to(diffusion.device)
    t = 0.16
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            idx = 0
            for _, train_data in enumerate(train_loader):
                idx += 1
                current_step += 1
                if current_step > n_iter:
                    break
                optimizer.zero_grad()
                y = diffusion(train_data['RI'])
                l_pix = loss_func(train_data['HI'],y)+t*dpss_func(train_data['HI'],y)
                # need to average in multi-gpu
                b, c, h, w = train_data['HI'].shape
                l_pix = l_pix.sum()/int(b*c*h*w)
                l_pix.backward()
                optimizer.step()

                # set log
                diffusion.log_dict['l_pix'] = l_pix.item()

                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        # tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            ri_img = diffusion(val_data['RI'])

            hr_img = Metrics.tensor2img(val_data['HI'])  # uint8
            lr_img = Metrics.tensor2img(val_data['LI'])  # uint8

            Metrics.save_img(Metrics.tensor2img(ri_img), '{}/{}_{}_ri_{}.npy'.format(result_path, current_step, idx, iter))
            
            Metrics.save_img(
                hr_img, '{}/{}_{}_hi.npy'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_li.npy'.format(result_path, current_step, idx))

            # generation
            eval_psnr = Metrics.calculate_psnr(ri_img, hr_img)
            eval_ssim = Metrics.calculate_ssim(ri_img, hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))




                
