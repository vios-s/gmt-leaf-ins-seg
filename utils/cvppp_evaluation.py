from detectron2.evaluation import DatasetEvaluator
import torch
import numpy as np
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.engine import DefaultPredictor

import os
from torchvision.io import read_image, ImageReadMode
from PIL import Image
import json

# custom utils
from .lsc_eval_tools import DiffFGLabels, BestDice, FGBGDice
# from .custom_coco_infer import custom_inference_on_dataset

def read_gt_mask(mask_name):
    if mask_name.endswith('.npy'):
        gt_mask = np.load(mask_name)
    else:
        # read mask with torch
        gt_mask = np.asarray(Image.open(mask_name).convert('L')) # looks ok for all the leaf datasets
    obj_ids = np.unique(gt_mask)
    if len(obj_ids) > 1:
        obj_ids = obj_ids[1:]
    else:
        raise ValueError('No leaves found')
    gt_masks = gt_mask == obj_ids[:, None, None]
    gt_masks = gt_masks.astype(np.uint8)

    return gt_masks

def filter_gt_mask(gt_masks, size_threshold=[0,1e8]):
    # mask size filtering
    size_filtered_indices = [False] * gt_masks.shape[0]
    for i in range(gt_masks.shape[0]):
        mask_size = np.count_nonzero(gt_masks[i])
        if mask_size > size_threshold[0] and mask_size <= size_threshold[1]:
            size_filtered_indices[i] = True
    size_filtered_masks = gt_masks[size_filtered_indices]

    gt_single_channel_mask = np.zeros((size_filtered_masks.shape[1], size_filtered_masks.shape[2]), dtype=np.uint8) # np (w, h)
    for i in range(size_filtered_masks.shape[0]):
        gt_single_channel_mask[size_filtered_masks[i] > 0] = i+1
    # gt_binary_mask = np.where(gt_single_channel_mask > 0, 255, 0).astype(np.uint8) # np (w, h)

    return gt_single_channel_mask

# w/ small objects
def filter_pred_mask(pred, score_threshold=0.0, size_threshold=[0,1e8]):
    pred_ins = pred[0]['instances'].to('cpu') # [0] here is problematic?
    pred_scores = pred_ins.scores
    pred_boxes = pred_ins.pred_boxes.tensor
    pred_masks = pred_ins.pred_masks

    # confidence score filtering
    filtered_indices = pred_scores >= score_threshold
    filtered_boxes = pred_boxes[filtered_indices]
    filtered_scores = pred_scores[filtered_indices]
    filtered_masks = pred_masks[filtered_indices] # np (w, h)

    # mask size filtering
    size_filtered_indices = [False] * filtered_masks.shape[0]
    for i in range(filtered_masks.shape[0]):
        mask_size = filtered_masks[i].count_nonzero().item()
        if mask_size > size_threshold[0] and mask_size <= size_threshold[1]:
            size_filtered_indices[i] = True

    size_filtered_boxes = filtered_boxes[size_filtered_indices]
    size_filtered_scores = filtered_scores[size_filtered_indices]
    size_filtered_masks = filtered_masks[size_filtered_indices]

    # combine multi-channel masks to single channel
    single_channel_mask = np.zeros((size_filtered_masks.shape[1], size_filtered_masks.shape[2]), dtype=np.uint8)
    for i in range(size_filtered_masks.shape[0]):
        single_channel_mask[size_filtered_masks[i].numpy() > 0] = i+1
    # binary_mask = np.where(single_channel_mask > 0, 255, 0).astype(np.uint8)

    return single_channel_mask

class CVPPPEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, score_threshold=0.0, 
                 all=[0,1e8], 
                #  large=[64**2,1e8], medium=[32**2, 64**2], small=[0,32**2], # cvppp_a1 and ko modify
                # large=[62**2,1e8], medium=[36**2, 62**2], small=[0,36**2], # cvppp_a1 raw
                #  large=[56**2,1e8], medium=[35**2, 56**2], small=[0,35**2], # ko raw
                 large=[24**2,1e8], medium=[12**2, 24**2], small=[0,12**2], # msu_pid modify
                #  large=[16**2,1e8], medium=[10**2, 16**2], small=[0,10**2], # msu_pid raw
                evaluate_sizes=['all', 'large', 'medium', 'small'], 
                logger=None):
        self._dataset_name = dataset_name
        self.cfg = cfg
        self.score_threshold = score_threshold
        self.logger = logger

        # size definition (left < s <= right)
        self.all = all
        self.large = large
        self.medium = medium
        self.small = small

        self.evaluate_sizes = evaluate_sizes
        # for saving best model
        # self.best_metric = -float('inf')
        # self.checkpointer = checkpointer

        self.reset()

    def reset(self):
        self.diff_fg = {size: [] for size in self.evaluate_sizes}
        self.dice = {size: [] for size in self.evaluate_sizes}
        self.sbd = {size: [] for size in self.evaluate_sizes}
        self.fgbg_dice = {size: [] for size in self.evaluate_sizes}

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            img_name = input["file_name"]
            if self.cfg.INPUT.DATASET_MAPPER_NAME == "cvppp_lsc":
                mask_name = img_name.replace('_rgb', '_label') # validation
                # mask_name = img_name.replace('_rgb', '_label').replace('.png', '.npy') # test
            elif self.cfg.INPUT.DATASET_MAPPER_NAME == "komatsuna_rgb":
                mask_name = img_name.replace('multi_plant', 'multi_label').replace('rgb_', 'label_')
            elif self.cfg.INPUT.DATASET_MAPPER_NAME == "msu_pid":
                mask_name = img_name.replace('Images', 'Labels').replace('_rgb.png', '_label_rgb.png')
            
            gt_masks = read_gt_mask(mask_name)

            # vis_metrics starts
            temp_results = {size: {} for size in self.evaluate_sizes}
            temp_results['img_name'] = img_name
            # vis_metrics ends
            
            # Calculate metrics on different sizes
            for size in self.evaluate_sizes:
                gt_label = filter_gt_mask(gt_masks, size_threshold=getattr(self, size))
                pred_label = filter_pred_mask([output], self.score_threshold, size_threshold=getattr(self, size))
            
                if np.max(gt_label) > 0: # filter out images without leaves
                    self.diff_fg[size].append(DiffFGLabels(pred_label,gt_label))
                    self.fgbg_dice[size].append(FGBGDice(pred_label,gt_label))
                    bd = BestDice(pred_label,gt_label)
                    bd2 = BestDice(gt_label,pred_label)
                    self.dice[size].append(bd) # this is like the precision (used in the original CVPPP competition)
                    self.sbd[size].append(min(bd, bd2))

            # vis_metrics starts
                    temp_results[size] = {"absdiffFG": str(np.abs(self.diff_fg[size][-1])),
                                          "FgBgDice": str(self.fgbg_dice[size][-1]),
                                          "bestDice": str(bd),
                                          "SBD": str(self.sbd[size][-1])}
                    
            with open(f'./metrics_for_vis_{self.cfg.INPUT.DATASET_MAPPER_NAME}.json', 'a') as f:
                json.dump(temp_results, f, indent=4)
            # vis_metrics ends

    def evaluate(self):
        # Finalize and compute metrics
        results = {size: {} for size in self.evaluate_sizes}
        for size in self.evaluate_sizes:
            if len(self.dice[size]) > 0:
                results[size] = {"diffFG_mean": np.mean(self.diff_fg[size]),
                                "diffFG_std": np.std(self.diff_fg[size]),
                                "absdiffFG_mean": np.mean(np.abs(self.diff_fg[size])),
                                "absdiffFG_std": np.std(np.abs(self.diff_fg[size])),
                                "bestDice_mean": np.mean(self.dice[size]),
                                "bestDice_std": np.std(self.dice[size]),
                                "FgBgDice_mean": np.mean(self.fgbg_dice[size]),
                                "FgBgDice_std": np.std(self.fgbg_dice[size]),
                                'SBD_mean': np.mean(self.sbd[size]),
                                'SBD_std': np.std(self.sbd[size]),
                                }
            else:
                results[size] = {"diffFG_mean": 0.,
                                "diffFG_std": 0.,
                                "absdiffFG_mean": 0.,
                                "absdiffFG_std": 0.,
                                "bestDice_mean": 0.,
                                "bestDice_std": 0.,
                                "FgBgDice_mean": 0.,
                                "FgBgDice_std": 0.,
                                'SBD_mean': 0.,
                                'SBD_std': 0.,
                                }

        if self.logger is not None:
            self.logger.log(results)
        
        # save best model
        # if results['all']['bestDice_mean'] > self.best_metric:
        #     self.best_metric = results['all']['bestDice_mean']
        #     self.checkpointer.save('model_bestdice')

        # Reset for next evaluation
        self.reset()

        return results
