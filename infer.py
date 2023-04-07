import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/test_256_90.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
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
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase, s='valid')
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LI=False)

        hi_img = Metrics.tensor2img(visuals['HI'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        ri_img_mode = 'grid'
        if ri_img_mode == 'single':
            # single img series
            ri_img = visuals['RI']  # uint8
            sample_num = ri_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(ri_img[iter]), '{}/{}_{}_ri_{}.npy'.format(result_path, current_step, idx, iter))
        else:
            # grid img
            ri_img = Metrics.tensor2img(visuals['RI'])  # uint8
            Metrics.save_img(
                ri_img, '{}/{}_{}_ri_process.npy'.format(result_path, current_step, idx))
            Metrics.save_img(
                Metrics.tensor2img(visuals['RI'][-1]), '{}/{}_{}_ri.npy'.format(result_path, current_step, idx))

        Metrics.save_img(
            hi_img, '{}/{}_{}_hi.npy'.format(result_path, current_step, idx))
        Metrics.save_img(
            fake_img, '{}/{}_{}_inf.npy'.format(result_path, current_step, idx))

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['RI'][-1]), hi_img)

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)
