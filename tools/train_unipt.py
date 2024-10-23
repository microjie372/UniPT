import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
# from pcdet.datasets import build_dataloader
# from pcdet.models import build_network, model_fn_decorator
from pcdet.datasets import build_dataloader_mdf, build_dataloader
from pcdet.models import build_network_multi_db, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
# from train_utils.train_utils import train_model
from train_utils.train_multi_db_utils import train_model

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--frozen_backbone', action='store_true', default=False, help='froze the backbone when training')
    parser.add_argument('--source_one_name', type=str, default="nusc", help='enter the name of the first dataset of merged datasets')
	
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    #parser.add_argument('--use_amp', action='store_true', default=False, help='use mix persion')

    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    #if args.use_amp:
    #    from pcdet import set_mode
    #    set_mode(half_mode=True)
    #    from pcdet import train_half_mode
    #    print(f"train_half_mode={train_half_mode}")

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    #args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()

    if args.source_one_name not in ["waymo", "nusc", "kitti"]:
        raise RuntimeError('Does not exist for source_one_name')

    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ps_label_dir = output_dir / 'ps_label'
    ps_label_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------

    logger.info('**********************Using Two DataLoader For Point Reconstruction**********************')
    logger.info('**********************VALUE of source_one_name= %s**********************' % args.source_one_name)

    source_set, source_loader, source_sampler = build_dataloader_mdf(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        drop_last=True,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )
    source_set_2, source_loader_2, source_sampler_2 = build_dataloader_mdf(
        dataset_cfg=cfg.DATA_CONFIG_SRC_2,
        class_names=cfg.DATA_CONFIG_SRC_2.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        drop_last=True,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

    # add the dataset_source flag into uni3d_norm layer, for training stage, we use the default value of 1
    if cfg.MODEL.get('POINT_T', None):
        cfg.MODEL.POINT_T.update({"db_source": 1})
    if cfg.MODEL.get('BACKBONE_3D', None):
        cfg.MODEL.BACKBONE_3D.update({"db_source": 1})
    if cfg.MODEL.get('DENSE_3D_MoE', None):
        cfg.MODEL.DENSE_3D_MoE.update({"db_source": 1})
    if cfg.MODEL.get('BACKBONE_2D', None):
        cfg.MODEL.BACKBONE_2D.update({"db_source": 1})
    if cfg.MODEL.get('DENSE_2D_MoE', None):
        cfg.MODEL.DENSE_2D_MoE.update({"db_source": 1})
    if cfg.MODEL.get('PFE', None):
        cfg.MODEL.PFE.update({"db_source": 1})

    # train_set, train_loader, train_sampler = build_dataloader(
    #     dataset_cfg=cfg.DATA_CONFIG,
    #     class_names=cfg.CLASS_NAMES,
    #     batch_size=args.batch_size,
    #     dist=dist_train, workers=args.workers,
    #     logger=logger,
    #     training=True,
    #     merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
    #     total_epochs=args.epochs,
    #     seed=666 if args.fix_random_seed else None
    # )

    model = build_network_multi_db(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), num_class_s2=len(cfg.DATA_CONFIG_SRC_2.CLASS_NAMES), \
         dataset=source_set, dataset_s2=source_set_2, source_one_name=args.source_one_name)
    # model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters

    if args.frozen_backbone:
        logger.info('**********************Note that Frozen Backbone: %s**********************')
        model.frozen_model(model)

    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    max_len_dataset = len(source_loader) if len(source_loader) > len(source_loader_2) else len(source_loader_2)
    total_iters_each_epoch = max_len_dataset if not args.merge_all_iters_to_one_epoch \
        else max_len_dataset // args.epochs

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=total_iters_each_epoch, total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    train_func = train_model
    # lr_scheduler, lr_warmup_scheduler = build_scheduler(
    #     optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
    #     last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    # )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_func(
        model,
        optimizer,
        source_loader,
        source_loader_2,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        ps_label_dir=ps_label_dir,
        source_sampler=source_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger,
        #use_amp=args.use_amp
    )

    if hasattr(source_set, 'use_shared_memory') and source_set.use_shared_memory:
        source_set.clean_shared_memory()

    # train_model(
    #     model,
    #     optimizer,
    #     train_loader,
    #     model_func=model_fn_decorator(),
    #     lr_scheduler=lr_scheduler,
    #     optim_cfg=cfg.OPTIMIZATION,
    #     start_epoch=start_epoch,
    #     total_epochs=args.epochs,
    #     start_iter=it,
    #     rank=cfg.LOCAL_RANK,
    #     tb_log=tb_log,
    #     ckpt_save_dir=ckpt_dir,
    #     train_sampler=train_sampler,
    #     lr_warmup_scheduler=lr_warmup_scheduler,
    #     ckpt_save_interval=args.ckpt_save_interval,
    #     max_ckpt_save_num=args.max_ckpt_save_num,
    #     merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch
    # )

    # if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
    #     train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()