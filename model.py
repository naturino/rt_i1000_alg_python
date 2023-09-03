import random
import cv2
import numpy as np
import json
import warnings
from mmdet.apis import init_detector, inference_detector
from mmpretrain.apis import ImageClassificationInferencer
from file_operate import *

# 忽略特定警告类别
warnings.filterwarnings("ignore", category=UserWarning, message="__floordiv__ is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid")

class MMClsModel:

    def __init__(self,config_file,checkpoint_file,device):
        self.model = ImageClassificationInferencer(config_file, pretrained=checkpoint_file, device=device)

    def infer(self,img_file):
        results = self.model(img_file)
        print()
        return results

    def preprocess(self,img):
        return img
    def postprocess(self,results):
        dst = []
        for result in results:
            del result['pred_scores']
            dst.append(result)
        return dst

    def save_json(self,results,save_path):

        directory_path = os.path.dirname(save_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        if save_path  != None:
            with open(save_path, 'w') as json_file:
                json.dump(results, json_file, indent=4)

class MMDetModel:

    def __init__(self,config_file,checkpoint_file,device):

        self.model = init_detector(config_file, checkpoint_file, device=device)  # or device='cuda:0'
        self.colors = {}

    def infer(self,img_file):
        self.result = inference_detector(self.model, img_file)
        return self.result

    def visual(self,image_path, results, save_floder = './vis/det',thikness  = 2):
        # 加载图像
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
        # 绘制结果
        for result in results:
            label = result['label']
            score = result['score']
            bbox = result['bbox']

            if label not in self.colors.keys():
                self.colors[label] = (random.randint(50, 200),random.randint(50, 200),random.randint(50, 200))
            color = self.colors[label]

            # 绘制矩形框
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thikness)

            # 在左上角绘制score
            cv2.putText(image, f"{label}:{score:.2f}", (x1, y1- thikness), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if not os.path.exists(save_floder):
            os.makedirs(save_floder)

        save_name = os.path.basename(image_path)
        save_path = os.path.join(save_floder,save_name)
        cv2.imwrite(save_path,image)
        # cv2.imencode('.jpg', image)[1].tofile(save_path)

    def postprocess(self,result,score_thresh):

        pred_instances = result.pred_instances.numpy().cpu()
        scores = pred_instances.scores
        bboxes=pred_instances.bboxes
        labels=pred_instances.labels
        dst = []
        for label,score,bbox in zip(labels,scores,bboxes):

            if float(score) < score_thresh:
                continue

            bbox = [int(x) for x in bbox.tolist()]
            score = round(float(score), 3)
            label = int(label)

            dst_dict = {
                'label':label,
                'score':score,
                'bbox':bbox,
            }
            dst.append(dst_dict)

        return dst

    def del_key(self,list_dict,key):
        dst = []
        for dict_info in list_dict:
            if key not in dict_info.keys():continue
            del dict_info[key]
            dst.append(dict_info)
        return dst

    def save_json(self,json_dict,save_path):

        if os.path.isfile(save_path):
            os.remove(save_path)

        directory_path = os.path.dirname(save_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        if save_path  != None:
            with open(save_path, 'w') as json_file:
                json.dump(json_dict, json_file, indent=4)

    def iou(self,obj1,obj2,iou_thresh):

        iou,_ = self.box_iou(obj1['bbox'], obj2['bbox'])

        return iou
    def box_area(self,box):
        x1_box, y1_box, x2_box, y2_box = box
        area = (x2_box - x1_box) * (y2_box - y1_box)
        return area

    def box_iou(self,box1, box2):
        # 提取box1和box2的坐标信息
        x1_box1, y1_box1, x2_box1, y2_box1 = box1
        x1_box2, y1_box2, x2_box2, y2_box2 = box2

        # 计算交集的坐标信息
        x_intersection = max(x1_box1, x1_box2)
        y_intersection = max(y1_box1, y1_box2)
        w_intersection = max(0, min(x2_box1, x2_box2) - x_intersection)
        h_intersection = max(0, min(y2_box1, y2_box2) - y_intersection)

        # 计算交集的面积
        area_intersection = w_intersection * h_intersection

        # 计算并集的面积
        area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
        area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
        area_union = area_box1 + area_box2 - area_intersection

        # 计算IoU
        iou = area_intersection / (area_union + 1e-6)  # 加上一个小的epsilon避免除零错误

        return iou,area_intersection

    def nms(self, dst, iou_threshold):
        if not dst:
            return []

        if iou_threshold < 0 or iou_threshold > 1:
            raise ValueError("iou_threshold  must be in the range [0, 1].")

        dst_sorted = sorted(dst, key=lambda x: x['score'], reverse=True)
        dst_filtered = []

        while len(dst_sorted) != 0:
            current_obj = dst_sorted.pop(0)
            dst_filtered.append(current_obj)
            if not dst_sorted:
                break
            iou_values = [self.iou(current_obj, obj ,iou_threshold) for obj in dst_sorted]

            indices_to_remove = []
            # for index, iou in enumerate(iou_values):
            #     if iou > iou_threshold:
            #         indices_to_remove.append(index)
            # for index in reversed(indices_to_remove):
            #     dst_sorted.pop(index)
            for index, iou in reversed(list(enumerate(iou_values))):
                if iou > iou_threshold:
                    dst_sorted.pop(index)
        return dst_filtered

class MMInstSegModel(MMDetModel):

    def visual(self,image_path, results,save_floder = './vis/det',thikness  = 2):
        # 加载图像
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
        # 绘制结果
        for result in results:
            label = result['label']
            score = result['score']
            bbox = result['bbox']
            cnts = [np.array(cnt) for cnt in result['cnts']]
            mask = result['mask']

            if label not in self.colors.keys():
                self.colors[label] = (random.randint(50, 200),random.randint(50, 200),random.randint(50, 200))
            color = self.colors[label]

            # 绘制mask区域（半透明）
            mask_indices = np.where(mask)
            mask_image = image.copy()
            mask_image[mask_indices[0], mask_indices[1]] = color
            alpha = 0.2 # 设置mask的透明度，值越大越不透明，值为1时完全不透明

            cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0, image)
            cv2.drawContours(image, cnts, -1, self.colors[label], thikness)

            # 绘制矩形框
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thikness)

            # 在左上角绘制score
            cv2.putText(image, f"{label}:{score:.2f}", (x1, y1- thikness), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if not os.path.exists(save_floder):
            os.makedirs(save_floder)

        save_name = os.path.basename(image_path)
        save_path = os.path.join(save_floder,save_name)
        cv2.imencode('.png', image)[1].tofile(save_path)

    def max_cnts(self,cnts,min_area):
        if len(cnts) > 1:
            max_area = 0
            max_contour = np.array([])
            # 遍历所有轮廓
            for contour in cnts:
                area = cv2.contourArea(contour)
                if area < min_area: continue
                if area > max_area:
                    max_area = area
                    max_contour = contour
            cnts = [max_contour]
        return cnts

    def postprocess(self,result,score_thresh,min_area):

        pred_instances = result.pred_instances.numpy().cpu()
        masks = pred_instances.masks
        scores = pred_instances.scores
        bboxes=pred_instances.bboxes
        labels=pred_instances.labels
        dst = []
        masks = np.array(masks,dtype=np.int16)
        for label,score,bbox,mask in zip(labels,scores,bboxes,masks):

            if float(score) < score_thresh:
                continue

            _,mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
            mask = mask.astype(np.uint8)
            cnts,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cnts = self.max_cnts(cnts,min_area)
            if len(cnts) < 1:continue
            if cnts[0].shape[0] < 5: continue

            bbox = [int(x) for x in bbox.tolist()]
            score = round(float(score), 3)
            cnts_dst = [cnt.reshape((-1,2)).tolist() for cnt in cnts]
            label = int(label)

            dst_dict = {
                'label':label,
                'score':score,
                'bbox':bbox,
                'cnts':cnts_dst,
                'mask':mask,
                'cnts_src': cnts,
            }
            dst.append(dst_dict)

        return dst

    def iou(self,obj1,obj2,iou_thresh):

        iou,inter_area = self.box_iou(obj1['bbox'], obj2['bbox'])
        min_box_area = min(self.box_area(obj1['bbox']),self.box_area(obj2['bbox']))
        if iou > iou_thresh or inter_area > min_box_area * 0.8:
            iou,inter_mask_area = self.mask_iou(obj1['mask'], obj2['mask'])
            min_mask_area = min(np.count_nonzero(obj1['mask']),np.count_nonzero(obj2['mask']))
            if  inter_mask_area > min_mask_area * 0.8:
                iou = inter_mask_area/(min_mask_area + 1e-6)
        return iou

    def mask_iou(self, mask1, mask2):
        # 添加合法性检查，确保输入参数是布尔类型的 Numpy 数组
        if not isinstance(mask1, np.ndarray) or not isinstance(mask2, np.ndarray):
            raise ValueError("mask1 and mask2 must be Numpy arrays.")

        # Calculate the intersection over union (IoU) of two masks
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        iou = intersection / (union + 1e-6)  # Add a small epsilon to avoid division by zero

        return iou,intersection

    def nms(self, dst, iou_threshold):
        if not dst:
            return []

        if iou_threshold < 0 or iou_threshold > 1:
            raise ValueError("iou_threshold  must be in the range [0, 1].")

        dst_sorted = sorted(dst, key=lambda x: x['score'], reverse=True)
        dst_filtered = []

        while len(dst_sorted) != 0:
            current_obj = dst_sorted.pop(0)
            dst_filtered.append(current_obj)
            if not dst_sorted:
                break
            iou_values = [self.iou(current_obj, obj ,iou_threshold) for obj in dst_sorted]

            indices_to_remove = []
            # for index, iou in enumerate(iou_values):
            #     if iou > iou_threshold:
            #         indices_to_remove.append(index)
            # for index in reversed(indices_to_remove):
            #     dst_sorted.pop(index)
            for index, iou in reversed(list(enumerate(iou_values))):
                if iou > iou_threshold:
                    dst_sorted.pop(index)
        return dst_filtered
