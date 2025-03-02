# modified by Feng Chen from https://github.com/facebookresearch/Mask2Former
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
# Suppress specific UserWarning
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import warnings
warnings.simplefilter("ignore", category=UserWarning)

# main lib here
import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    hooks,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, DatasetMapper
from detectron2.data import transforms as T

# customise libs
from utils.cvppp_evaluation import CVPPPEvaluator
from utils.custom_coco_eval import CustomCOCOEvaluator
from utils.register_datasets import register_datasets

import wandb

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    def build_hooks(self):
        ret = super().build_hooks() # inherit the hooks from the default hooks
        if self.cfg.INPUT.DATASET_MAPPER_NAME == "cvppp_lsc" or "komatsuna_rgb" or "msu_pid":
            ret.insert(-1, hooks.BestCheckpointer(eval_period=self.cfg.TEST.EVAL_PERIOD, 
                            checkpointer=self.checkpointer,
                            val_metric='all/bestDice_mean',
                            mode='max',
                            file_prefix='model_best_bd',
                            ))
            ret.insert(-1, hooks.BestCheckpointer(eval_period=self.cfg.TEST.EVAL_PERIOD, 
                                        checkpointer=self.checkpointer,
                                        val_metric='all/SBD_mean',
                                        mode='max',
                                        file_prefix='model_best_sbd',
                                        ))
        elif self.cfg.INPUT.DATASET_MAPPER_NAME == "coco_person":
            ret.insert(-1, hooks.BestCheckpointer(eval_period=self.cfg.TEST.EVAL_PERIOD, 
                                                checkpointer=self.checkpointer,
                                                val_metric='segm/AP',
                                                mode='max',
                                                file_prefix='model_best_ap',
                                                ))
        return ret
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        if cfg.INPUT.DATASET_MAPPER_NAME == "cvppp_lsc" or "komatsuna_rgb" or "msu_pid":
            augmentation = [T.Resize((cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE), interp=2),
                            ]
            mapper = DatasetMapper(is_train=False,
                                   augmentations=augmentation,
                                   image_format=cfg.INPUT.FORMAT,
                                #    instance_mask_format = "bitmask",
                                #    use_instance_mask = True,
                                #    recompute_boxes = True,
                                   )
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_person":
            augmentation = [T.Resize((cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE), interp=2),
                            ]
            mapper = DatasetMapper(is_train=False,
                                   augmentations=augmentation,
                                   image_format=cfg.INPUT.FORMAT,
                                   )
        # else:
        #     mapper = None
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []

        # instance segmentation
        if cfg.INPUT.DATASET_MAPPER_NAME == "cvppp_lsc" or "komatsuna_rgb" or "msu_pid":
            if cfg.MODEL.MASK_FORMER.LOGGER is None:
                logger = None
            else:
                logger = wandb
            # evaluator_list.append(CVPPPEvaluator(dataset_name, cfg, score_threshold=0.8, logger=logger,
            #                                      large=[64**2,1e8], medium=[32**2, 64**2], small=[0,32**2]))
            evaluator_list.append(CVPPPEvaluator(dataset_name, cfg, score_threshold=0.85, logger=logger,))
        # elif cfg.INPUT.DATASET_MAPPER_NAME == "msu_pid":
        #     if cfg.MODEL.MASK_FORMER.LOGGER is None:
        #         logger = None # for offline test
        #     else:
        #         logger = wandb
        #     evaluator_list.append(CVPPPEvaluator(dataset_name, cfg, score_threshold=0.8, logger=logger, 
        #                                          large=[24**2,1e8], medium=[12**2, 24**2], small=[0,12**2])) # only definition of size changes
            # evaluator_list.append(CustomCOCOEvaluator(dataset_name, output_dir=output_folder, distributed=False, allow_cached_coco=False))
            # evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, distributed=False, allow_cached_coco=False)))
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_person":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, distributed=False, allow_cached_coco=False))
    
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        
        # coco person
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_person":
            # check min_scale=cfg.INPUT.MIN_SCALE, max_scale=cfg.INPUT.MAX_SCALE
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        
        # cvppp ins seg
        elif cfg.INPUT.DATASET_MAPPER_NAME == "cvppp_lsc" or "komatsuna_rgb":
            augmentation = [T.RandomFlip(horizontal=True, vertical=False),
                            T.RandomFlip(horizontal=False, vertical=True),
                            T.ResizeScale(min_scale=cfg.INPUT.MIN_SCALE, max_scale=cfg.INPUT.MAX_SCALE, target_height=cfg.INPUT.IMAGE_SIZE, target_width=cfg.INPUT.IMAGE_SIZE,), # it keeps the asptect ratio of the original image but do zoom-in and out
                            T.FixedSizeCrop(crop_size=(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE), pad=False), # if pad is False, only when crop_size < input image size will perform random crop
                            ]
            mapper = DatasetMapper(is_train=True,
                                   augmentations=augmentation,
                                   image_format=cfg.INPUT.FORMAT,
                                   instance_mask_format = "bitmask",
                                   use_instance_mask = True,
                                #    recompute_boxes = True, # we don't use boxes here
                                   )
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "msu_pid":
            augmentation = [
                            T.RandomFlip(horizontal=True, vertical=False),
                            T.RandomFlip(horizontal=False, vertical=True),
                            T.Resize((cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE), interp=2),
                            T.ResizeScale(min_scale=cfg.INPUT.MIN_SCALE, max_scale=cfg.INPUT.MAX_SCALE, target_height=cfg.INPUT.IMAGE_SIZE, target_width=cfg.INPUT.IMAGE_SIZE,), # it keeps the asptect ratio of the original image but do zoom-in and out
                            T.FixedSizeCrop(crop_size=(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE), pad=False), # if pad is False, only when crop_size < input image size will perform random crop
                            ]
            mapper = DatasetMapper(is_train=True,
                                   augmentations=augmentation,
                                   image_format=cfg.INPUT.FORMAT,
                                   instance_mask_format = "bitmask",
                                   use_instance_mask = True,
                                #    recompute_boxes = True, # we don't use boxes here
                                   )
            return build_detection_train_loader(cfg, mapper=mapper)
        # else:
        #     mapper = None
        #     return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    # we don't use TTA
    # @classmethod
    # def test_with_TTA(cls, cfg, model):
    #     logger = logging.getLogger("detectron2.trainer")
    #     # In the end of training, run an evaluation with TTA.
    #     logger.info("Running inference with test-time augmentation ...")
    #     model = SemanticSegmentorWithTTA(cfg, model)
    #     evaluators = [
    #         cls.build_evaluator(
    #             cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
    #         )
    #         for name in cfg.DATASETS.TEST
    #     ]
    #     res = cls.test(cfg, model, evaluators)
    #     res = OrderedDict({k + "_TTA": v for k, v in res.items()})
    #     return res

def setup(args):
    """
    Create configs and perform basic setups.
    """
    # dataset registration
    data_root = './datasets'
    
    # komatsuna_rgb train, val, test
    register_datasets(json_root=os.path.join(data_root, 'komatsuna_multi_view/multi_plant'),
                      json_paths=['komatsuna_rgb_train_coco_0.json', 'komatsuna_rgb_val_coco_0.json', 'komatsuna_rgb_test_coco_0.json',
                                  'komatsuna_rgb_train_coco_1.json', 'komatsuna_rgb_val_coco_1.json', 'komatsuna_rgb_test_coco_1.json',
                                  'komatsuna_rgb_train_coco_2.json', 'komatsuna_rgb_val_coco_2.json', 'komatsuna_rgb_test_coco_2.json',],
                      image_root=os.path.join(data_root, 'komatsuna_multi_view/multi_plant'),)

    # msu_pid train, val, test
    register_datasets(json_root=os.path.join(data_root, 'msu_pid/Release/Dataset/Images/Arabidopsis'),
                      json_paths=['msu_pid_arabidopsis_train_coco_0.json', 'msu_pid_arabidopsis_val_coco_0.json', 'msu_pid_arabidopsis_test_coco_0.json',
                                  'msu_pid_arabidopsis_train_coco_1.json', 'msu_pid_arabidopsis_val_coco_1.json', 'msu_pid_arabidopsis_test_coco_1.json',
                                  'msu_pid_arabidopsis_train_coco_2.json', 'msu_pid_arabidopsis_val_coco_2.json', 'msu_pid_arabidopsis_test_coco_2.json',],
                      image_root=os.path.join(data_root, 'msu_pid/Release/Dataset/Images/Arabidopsis'),)
    
    # CVPPP A1 train and val
    register_datasets(json_root=os.path.join(data_root, 'cvppp2017/A1'),
                      json_paths=['A1_90_coco.json',
                                  'A1_10_coco.json',],
                      image_root=os.path.join(data_root, 'cvppp2017/A1'),)

    # # CVPPP A1 test [you need to download the online official test set]
    # register_datasets(json_root=os.path.join(data_root, 'CVPPP2017_Andrei/CVPPP_2017_original/CVPPP2017_testing/testing/A1'),
    #                   json_paths=['A1_coco.json',],
    #                   image_root=os.path.join(data_root, 'CVPPP2017_Andrei/CVPPP_2017_original/CVPPP2017_testing/testing/A1'),)
    
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg) # here we have some custom configs for harmonic embeddings
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg

def main(args):
    cfg = setup(args)
    display_name = os.path.splitext(os.path.split(args.config_file)[-1])[0]
    wandb.init(project='GuideM2F', name=display_name, config=cfg, sync_tensorboard=True)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.dist_url = None # we are not distributed
    # args.resume=True
    print("Command Line Args:", args)
    main(args)
    
    # at the end of training, perform testing (only for komatsuna and msu_pid
    # for cvppp lsc 2017, please follow the instructions at official website
    # to test on the hidden test set)
    args.eval_only = True
    cfg = setup(args)
    cfg.defrost()
    cfg.DATASETS.TEST = (cfg.DATASETS.TEST[0].replace('_val_','_test_'),)
    cfg.freeze()

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(os.path.join(cfg.OUTPUT_DIR, "model_best_bd.pth"), resume=False)
    model.eval()
    res = Trainer.test(cfg, model)
    print('test_best_bd:', res['all'])

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(os.path.join(cfg.OUTPUT_DIR, "model_best_sbd.pth"), resume=False)
    model.eval()
    res = Trainer.test(cfg, model)
    print('test_best_sbd:', res['all'])

    # for distributed training
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     # dist_url=args.dist_url,
    #     dist_url = None,
    #     args=(args,),
    # )
