import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from guide_train_net import Trainer, setup
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    hooks,
)

from detectron2.checkpoint import DetectionCheckpointer
import wandb
import matplotlib.pyplot as plt

from utils.cvppp_evaluation import read_gt_mask
from torchvision.io import read_image
from detectron2.utils.visualizer import Visualizer, VisImage, ColorMode
import h5py

def main(config_files):
    # selected_model_list = ['best_bd']
    selected_model_list = ['best_sbd']
    for selected_model in selected_model_list:
        avg_res = {size: {'bestDice_mean': [],
                        'bestDice_std': 0., 
                        'SBD_mean': [],
                        'SBD_std': 0., 
                        'absdiffFG_mean': [],
                        'absdiffFG_std': 0.,} 
                for size in ['all', 'large', 'medium', 'small']}
        
        data_split_idx_str = ''
        for run_idx, config_file in enumerate(config_files):
            data_split_idx_str = data_split_idx_str + config_file.split('/')[-1][-6:-5]
            device = 'cuda'
            args = default_argument_parser().parse_args()
            args.eval_only = True
            args.config_file = config_file
            print("Command Line Args:", args)
            cfg = setup(args)

            # change cfg.test_datasets to test set
            cfg.defrost()
            cfg.DATASETS.TEST = (cfg.DATASETS.TEST[0].replace('_val_','_test_'),)
            cfg.MODEL.MASK_FORMER.LOGGER = None
            cfg.freeze()

            # build and load model
            model = Trainer.build_model(cfg) # when building, already to cfg.MODEL.DEVICE (default='cuda')
            # Note: resume = True will load the latest checkpoint, not best
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(os.path.join(cfg.OUTPUT_DIR, f"model_{selected_model}.pth"), resume=False)
            model.eval()
            res = Trainer.test(cfg, model)
            print(res)
            # save individual results
            save_json_file_name = config_file.split('/')[-1][:-5] + f'_{selected_model}.json'
            with open(os.path.join('./submission/wacv_with_size',save_json_file_name), 'w') as f:
                json.dump(res, f)

            for size in ['all', 'large', 'medium', 'small']: # we only care sbd_mean, bd_mean, and absdiff_mean
                for metric in ['bestDice_mean', 'SBD_mean', 'absdiffFG_mean']:
                    avg_res[size][metric].append(res[size][metric])

        for size in ['all', 'large', 'medium', 'small']:
            for metric in ['bestDice_mean', 'SBD_mean', 'absdiffFG_mean']:
                res_mean = np.mean(avg_res[size][metric])
                res_std = np.std(avg_res[size][metric])
                avg_res[size][metric] = res_mean
                avg_res[size][metric.replace('_mean','_std')] = res_std
        print(avg_res)
        
        # save avg results
        save_json_file_name = config_files[0].split('/')[-1][:-7] + f'_rpt{len(config_files)}_ds{data_split_idx_str}_{selected_model}.json' # rpt = repeated times, ds = data splits
        with open(os.path.join('./submission/wacv_with_size',save_json_file_name), 'w') as f:
            json.dump(avg_res, f)

if __name__ == "__main__":
    # msu_pid baseline
    cf = ['./configs/guide_exp/msu_pid/msu_pid_r50_baseline_cocopre_0.yaml',
          './configs/guide_exp/msu_pid/msu_pid_r50_baseline_cocopre_1.yaml',
          './configs/guide_exp/msu_pid/msu_pid_r50_baseline_cocopre_2.yaml',
           ]

    # msu_pid gmt
    cf = ['./configs/guide_exp/msu_pid/msu_pid_r50_guide_cocopre_0.yaml',
           './configs/guide_exp/msu_pid/msu_pid_r50_guide_cocopre_1.yaml',
           './configs/guide_exp/msu_pid/msu_pid_r50_guide_cocopre_2.yaml',
           ]

    # komatsuna baseline
    cf = ['./configs/guide_exp/komatsuna_rgb/komatsuna_rgb_r50_baseline_cocopre_0.yaml',
                    './configs/guide_exp/komatsuna_rgb/komatsuna_rgb_r50_baseline_cocopre_1.yaml',
                    './configs/guide_exp/komatsuna_rgb/komatsuna_rgb_r50_baseline_cocopre_2.yaml',
                    ]

    # komatsuna gmt
    cf = ['./configs/guide_exp/komatsuna_rgb/komatsuna_rgb_r50_guide_cocopre_0.yaml',
                    './configs/guide_exp/komatsuna_rgb/komatsuna_rgb_r50_guide_cocopre_1.yaml',
                    './configs/guide_exp/komatsuna_rgb/komatsuna_rgb_r50_guide_cocopre_2.yaml',
                    ]

    for config_files in [cf]:
        main(config_files)