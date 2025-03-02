import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

def register_datasets(json_root, json_paths, image_root):
    for json_path in json_paths:
        dataset_name = json_path.replace('.json','')
        if dataset_name not in DatasetCatalog.list():
            register_coco_instances(dataset_name, {}, os.path.join(json_root, json_path), image_root)