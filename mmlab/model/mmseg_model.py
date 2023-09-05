import os
import cv2
import torch
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

from mmengine import Config
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS
from mmseg.apis import show_result_pyplot as mmseg_visual
from mmseg.utils import dataset_aliases, get_classes, get_palette

class MMSegModel:

    def __init__(self,config_file,checkpoint_file,device):

        self.model = self.init_model(config_file, checkpoint_file, device=device)  # or device='cuda:0'
        self.test_pipeline = self.init_test_pipeline()
        self.classes = ['background']
        self.classes.extend(self.model.dataset_meta['classes'])

    def init_model(self, config,checkpoint, device: str = 'cuda:0',cfg_options=None):

        if isinstance(config, (str, Path)):
            config = Config.fromfile(config)

        elif not isinstance(config, Config):
            raise TypeError('config must be a filename or Config object, '
                            'but got {}'.format(type(config)))
        if cfg_options is not None:
            config.merge_from_dict(cfg_options)
        elif 'init_cfg' in config.model.backbone:
            config.model.backbone.init_cfg = None
        config.model.pretrained = None
        config.model.train_cfg = None
        init_default_scope(config.get('default_scope', 'mmseg'))

        model = MODELS.build(config.model)
        if checkpoint is not None:
            checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
            dataset_meta = checkpoint['meta'].get('dataset_meta', None)
            # save the dataset_meta in the model for convenience
            if 'dataset_meta' in checkpoint.get('meta', {}):
                # mmseg 1.x
                model.dataset_meta = dataset_meta
            elif 'CLASSES' in checkpoint.get('meta', {}):
                # < mmseg 1.x
                classes = checkpoint['meta']['CLASSES']
                palette = checkpoint['meta']['PALETTE']
                model.dataset_meta = {'classes': classes, 'palette': palette}
            else:
                warnings.simplefilter('once')
                warnings.warn(
                    'dataset_meta or class names are not saved in the '
                    'checkpoint\'s meta data, classes and palette will be'
                    'set according to num_classes ')
                num_classes = model.decode_head.num_classes
                dataset_name = None
                for name in dataset_aliases.keys():
                    if len(get_classes(name)) == num_classes:
                        dataset_name = name
                        break
                if dataset_name is None:
                    warnings.warn(
                        'No suitable dataset found, use Cityscapes by default')
                    dataset_name = 'cityscapes'
                model.dataset_meta = {
                    'classes': get_classes(dataset_name),
                    'palette': get_palette(dataset_name)
                }
        model.cfg = config  # save the config in the model for convenience
        model.to(device)
        model.eval()
        return model

    def init_test_pipeline(self):
        # prepare data
        cfg = self.model.cfg
        for t in cfg.test_pipeline:
            if t.get('type') == 'LoadAnnotations':
                cfg.test_pipeline.remove(t)
        # a pipeline for each inference
        pipeline = Compose(cfg.test_pipeline)
        return pipeline

    def infer(self,imgs):
        is_batch = True
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
            is_batch = False
        data = defaultdict(list)
        for img in imgs:
            if isinstance(img, np.ndarray):
                data_ = dict(img=img)
            else:
                data_ = dict(img_path=img)
            data_ = self.test_pipeline(data_)
            data['inputs'].append(data_['inputs'])
            data['data_samples'].append(data_['data_samples'])

        # forward the model
        with torch.no_grad():
            results = self.model.test_step(data)

        return results if is_batch else results[0]

    def visual(self, results, save_floder):

        image_path = results.img_path

        if not os.path.exists(save_floder):
            os.makedirs(save_floder)

        save_name = os.path.basename(image_path)
        save_path = os.path.join(save_floder, save_name)

        mmseg_visual(
            self.model,
            image_path,
            results,
            draw_gt=False,
            show=False,
            out_file=save_path)

    def postprocess(self,result,score_thresh):
        pass

    def get_masks(self,results,mask_value = 255):
        masks = {}
        image_path = results.img_path
        image = cv2.imread(image_path)

        if image is None:
            return masks

        sem_seg = results.pred_sem_seg.cpu().data
        num_classes = len(self.classes)
        ids = np.unique(sem_seg)[::-1]
        legal_indices = ids < num_classes
        ids = ids[legal_indices]
        labels = np.array(ids, dtype=np.int64)

        for label in labels:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[sem_seg[0] == label] = mask_value
            masks[self.classes[label]] = mask

        return masks

    def save_masks(self,results,save_path):

        masks = self.get_masks(results)

        if len(masks) == 0:
            return

        image_path = results.img_path
        img_file = os.path.basename(image_path)
        name, extension = os.path.splitext(img_file)
        save_floder = os.path.join(save_path, name)
        if not os.path.exists(save_floder):
            os.makedirs(save_floder)

        for label, mask in masks.items():
            mask_path = os.path.join(save_floder,f'{label}.png')
            cv2.imwrite(mask_path,mask)