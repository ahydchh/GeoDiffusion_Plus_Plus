import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset
from .coco import CocoDataset
from mmdet.datasets.builder import DATASETS

@DATASETS.register_module()
class NuImageDataset(CocoDataset):
    CLASSES = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]