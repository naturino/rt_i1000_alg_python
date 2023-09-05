import os
import time
import cv2
import json
import warnings
import numpy as np

from base_cv_alg import BasicCVAlg
from cut_cnt_patch import CutCntPatch
from file_operate import FileOperate

from mmlab.model.mmseg_model import MMSegModel
from mmlab.model.mmdet_model import MMDetModel
from mmlab.model.mmdet_model import MMInstSegModel
from mmlab.model.mmpre_model import MMClsModel

# 忽略特定警告类别
warnings.filterwarnings("ignore", category=UserWarning)

class I1000InstSegModel(MMInstSegModel):
    def __init__(self, config, logger):

        self.logger = logger
        temp_path = config['TEMP_PATH']
        self.save_img_floder = os.path.join(temp_path, 'vis', 'seg').replace('\\', '/')
        self.save_json_floder = os.path.join(temp_path, 'json', 'seg').replace('\\', '/')

        # 初始化变量
        self.empty = config['EMPTY']
        self.file_type = config['IMAGE_TYPE']
        self.segcv_post = config['SEGCV_POST']
        self.min_area = 400
        self.save_vis = config['SEG100X_SAVE_VIS']
        self.conf_score = config['SEG100X_CONFIDENCE_SCORE']
        self.iou_score = config['SEG100X_IOU_THRESH']

        # 初始化模型
        self.model = MMInstSegModel(config['SEG100X_MODEL_CONFIG'], config['SEG100X_CHECKPOINT'],
                                           config['DEVICE'])

        self.basicCVAlg = BasicCVAlg()
        self.fileOperate = FileOperate()

    def det_miss(self,img_path,results):
        img = cv2.imread(img_path,0)

        if img is None:
            return results

        cnts = self.basicCVAlg.cntsAlg.findContoursByImage(img,False)

        if cnts == []:
            return results

        add_mask = np.zeros(img.shape)
        cv2.drawContours(add_mask,cnts,-1,255,-1)

        areas = []
        mean_colors = []
        for result in results:
            result_cnts = result['cnts_src']
            cv2.drawContours(add_mask,result_cnts,-1,0,-1)

            # 统计面积
            area = cv2.contourArea(result_cnts[0])
            areas.append(area)

            # 统计颜色
            mask = result['mask']
            mean_color = self.basicCVAlg.cntsAlg.maskMean(img,mask)
            mean_colors.append(mean_color)

        # 进行开运算
        kernel = np.ones((5, 5), np.uint8)
        add_mask = cv2.morphologyEx(add_mask, cv2.MORPH_OPEN, kernel)
        add_mask = add_mask.astype(np.uint8)
        add_cnts, _ = cv2.findContours(add_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 根据面积进行过滤
        min_area, max_area = min(areas),max(areas)
        add_cnts = self.basicCVAlg.cntsAlg.filterByArea(add_cnts, min_area, max_area)

        # 根据颜色进行过滤
        min_color, max_color = min(mean_colors),max(mean_colors)
        add_cnts = self.basicCVAlg.cntsAlg.filterByColor(img, add_cnts, min_color, max_color)

        for cnt in add_cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            mask = np.zeros(img.shape)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            cnt_dst = [cnt.reshape((-1, 2)).tolist()]
            dst_dict = {
                'label':0,
                'score':0,
                'bbox':[x,y,x+w,y+h],
                'cnts':cnt_dst,
                'cnts_src': cnts,
                'mask':mask
            }
            results.append(dst_dict)
        return results

    def run(self, img_path, image = np.array([])):
        img_path = img_path.replace('\\', '/')
        self.logger.info(f'Seg Infer {img_path}')

        start_time = time.time()

        # 分割染色体
        # if image.size != 0:
        #     infer_result = self.model.infer(image)
        # else:
        infer_result = self.model.infer(img_path)

        infer_result = self.model.postprocess(infer_result, self.conf_score, self.min_area)

        if self.segcv_post:
            infer_result = self.det_miss(img_path, infer_result)

        # NMS去重
        infer_result = self.model.nms(infer_result, self.iou_score)

        if self.save_vis:
            img_file = os.path.basename(img_path)
            name, extension = os.path.splitext(img_file)
            save_patch_path = os.path.join(self.save_img_floder, name)
            self.fileOperate.mkdirs(save_patch_path)
            self.model.visual(img_path, infer_result, save_patch_path)

        infer_result = self.model.del_key(infer_result, 'mask')
        infer_result = self.model.del_key(infer_result, 'cnts_src')
        end_time = time.time()
        infer_time = round(end_time - start_time, 3)
        self.logger.info(f'Seg successufl {len(infer_result)} objs, cost {infer_time}s')
        return infer_result

    def chrom_score(self,result):
        score_value = max(0, 46 - len(result))
        score_cls = 0
        if score_value > 20:
            score_cls = 1
        elif score_value > 15:
            score_cls = 2
        elif score_value > 10:
            score_cls = 3
        elif score_value > 3:
            score_cls = 4
        return score_cls,score_value

class I1000ClsModel(MMClsModel):
    def __init__(self, config, logger,flip_model):

        self.logger = logger
        temp_path = config['TEMP_PATH']
        self.save_json_floder = os.path.join(temp_path, 'json', 'cls').replace('\\', '/')
        self.save_img_floder = os.path.join(temp_path, 'vis', 'cls').replace('\\', '/')
        # 初始化变量
        self.empty = config['EMPTY']
        self.file_type = config['IMAGE_TYPE']

        self.classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                        '21', '22', 'X', 'Y', 'None']



        self.cls_num = len(self.classes)
        self.save_vis = config['CLS100X_SAVE_VIS']
        self.img_max_size = config['CLS100X_MAXSIZE']

        # 初始化模型
        self.model = MMClsModel(config['CLS100X_MODEL_CONFIG'], config['CLS100X_CHECKPOINT'],
                                        config['DEVICE'])
        self.model_flip100x = flip_model
        self.ccp = CutCntPatch()
        self.fileOperate = FileOperate()
    def __count_by_cls(self, results):
        # 按照 'cls_class' 分组，并按照 'cls_score' 降序排序
        grouped_data = {}
        for entry in results:
            cls_class = entry['cls_class']
            if cls_class not in grouped_data:
                grouped_data[cls_class] = []
            grouped_data[cls_class].append(entry)

        # 每个分组按照 'cls_score' 降序排序
        for cls_class, group in grouped_data.items():
            grouped_data[cls_class] = sorted(group, key=lambda x: x['cls_score'], reverse=True)

        return grouped_data

    def __find_closest_value(self, none_cls_objs, target, diff_percent=0.1):
        """
        在给定对象列表中，查找与目标值最接近的对象尺寸及其对应的对象。

        参数：
        none_cls_objs (list): 包含多个对象的列表，每个对象包含 'crop_patch' 键用于提取尺寸信息。
        target (int): 目标尺寸，函数将查找最接近此尺寸的对象。
        diff_percent (float, 可选): 允许的最大差异百分比，默认为 0.1。

        返回：
        closest_value (int): 最接近目标尺寸的对象的尺寸。
        closest_obj (object): 对应最接近尺寸的对象。

        函数会遍历给定对象列表，计算每个对象的尺寸与目标尺寸之间的差值，然后根据允许的差异百分比和最小差异值，选取最接近的对象尺寸及其对象。
        """
        # 计算最大允许差值
        max_difference = target * diff_percent
        # 初始化变量以记录最接近的值和对应的对象
        closest_value = None
        closest_obj = ''
        # 初始化最小差值为正无穷大
        min_difference = float('inf')

        # 遍历输入的对象列表
        for none_cls_obj in none_cls_objs:
            # 获取当前对象的最大尺寸
            none_cls_obj_max_length = max(none_cls_obj['crop_patch'].shape)
            # 计算当前对象尺寸与目标尺寸的差值
            difference = abs(none_cls_obj_max_length - target)
            # 检查差值是否在允许范围内，并且比已记录的最小差值小
            if difference <= max_difference and difference < min_difference:
                # 更新记录的最接近值和对象
                closest_value = none_cls_obj_max_length
                closest_obj = none_cls_obj
                min_difference = difference

        # 返回最接近的值和对应的对象
        return closest_value, closest_obj

    def __update_zero_values(self, dictionary):
        """
        根据两侧的非零值，更新字典中数值为0的数据。

        参数：
        dictionary (dict): 需要进行零值补全的字典。

        返回：
        dst (dict): 包含补全后的零值的新字典。

        该函数用于对给定字典中数值为0的项进行补全，以其左侧和右侧非零值为依据。首先，函数会遍历字典键值，定位数值为0的键。然后，根据该键的位置，
        寻找左侧和右侧的非零值，并计算补全的零值序列。最终，函数会返回一个新的字典，其中包含了进行零值补全后的结果。
        """
        dst = {}
        keys = sorted(dictionary.keys())

        for i, key in enumerate(keys):
            if dictionary[key] == 0:
                left_val = None
                right_val = None
                left_idx = i - 1
                right_idx = i + 1

                if i == 0:
                    val = dictionary[key]
                    while right_idx < len(keys):
                        if dictionary[keys[right_idx]] != 0:
                            val = dictionary[keys[right_idx]]
                            break
                        right_idx += 1
                    dictionary[key] = val
                    dst[key] = val

                if i == len(keys) - 1:
                    val = dictionary[key]
                    while left_idx >= 0:
                        if dictionary[keys[left_idx]] != 0:
                            val = dictionary[keys[left_idx]]
                            break
                        left_idx -= 1
                    dictionary[key] = val
                    dst[key] = val

                while left_idx >= 0:
                    if dictionary[keys[left_idx]] != 0:
                        left_val = dictionary[keys[left_idx]]
                        break
                    left_idx -= 1

                while right_idx < len(keys):
                    if dictionary[keys[right_idx]] != 0:
                        right_val = dictionary[keys[right_idx]]
                        break
                    right_idx += 1

                if left_val is not None and right_val is not None:
                    count_zeros = right_idx - left_idx - 1
                    increment = (right_val - left_val) / (count_zeros + 1)
                    num = 0
                    for j in range(left_idx + 1, right_idx):
                        num += 1
                        dictionary[keys[j]] = left_val + increment * num
                        dst[keys[j]] = left_val + increment * num

        return dst

    def __estimate_length(self, grouped_data, no_contain):
        """
        估计每个类别的平均长度。

        参数：
        grouped_data (dict): 分组后的数据，按类别划分并包含相关信息的字典。
        no_contain (list): 不包含在估计中的类别列表。

        返回：
        grouped_length_update (dict): 更新后的每个类别的平均长度字典。

        该函数用于计算每个类别的平均长度，以及处理可能存在的零值情况。首先，函数会遍历分组数据，对于每个类别，计算其中样本的最大长度。
        如果类别内有两个样本，则计算这两个样本的最大长度平均值。
        最终，函数会调用 __update_zero_values 函数来处理可能的零值情况，确保每个类别都有一个估计的平均长度值。
        """
        # 计算每类的平均长度
        grouped_length = {}
        for i in range(self.cls_num - 3):
            grouped_length[i] = 0

        for cls_class, group in grouped_data.items():
            if cls_class in no_contain: continue
            label = group[0]['label']
            group_max_length = max(group[0]['crop_patch'].shape)
            if len(group) == 2:
                group0_max_length = max(group[0]['crop_patch'].shape)
                group1_max_length = max(group[1]['crop_patch'].shape)
                group_max_length = int((group0_max_length + group1_max_length) / 2)
            grouped_length[label] = group_max_length
        grouped_length_update = self.__update_zero_values(grouped_length)
        return grouped_length_update

    def modify_cls(self, results):
        """
        根据一系列规则对类别进行调整。

        参数：
        results (list): 包含数据信息的列表，用于调整和估计。

        返回：
        dst (list): 调整后的包含数据信息的列表。

        该函数用于根据一定规则对给定的数据进行类别调整和长度估计。它分为三个主要步骤：首先，保留每个分组的前两个条目，将多余的条目的 'cls_class' 设置为 none_cls；其次，对只有一条数据的类别进行匹配，尽量使其长度与其他类别相似；最后，对没有染色体的组进行长度估计，并将其分配到合适的类别中。

        步骤 1：保留每个分组的前两个条目，并将多余的条目的 'cls_class' 设置为 none_cls。

        步骤 2：对只有一条数据的类别进行匹配，匹配最相似且长度差小于指定百分比的类别。特殊情况是，对于 Y 染色体和在有 Y 染色体时的 X 染色体，只保留一条数据。如果没有匹配到相似类别，则不进行处理。

        步骤 3：对没有染色体的组，通过估计长度再分配到合适的类别中。估计长度时排除 X、Y 和 none_cls 类别，然后根据估计长度匹配合适的类别。

        最终，返回经过调整和估计后的数据列表。

        """
        max_bais_percent = 0.2
        none_cls_index = 24
        none_cls_name = self.classes[none_cls_index]
        grouped_data1 = self.__count_by_cls(results)  # 按照类别统计

        # 统计X和Y染色体数量
        num_Y = 0
        num_X = 0
        if 'Y' in grouped_data1.keys():
            num_Y = len(grouped_data1['Y'])
        if 'X' in grouped_data1.keys():
            num_X = len(grouped_data1['X'])

        keep2_per_cls = []
        for cls_class, group in grouped_data1.items():
            if len(group) > 2:
                keep2_per_cls.extend(group[:2])
                for extra_entry in group[2:]:
                    extra_entry['src_cls_class'] = extra_entry['cls_class']
                    extra_entry['src_label'] =  extra_entry['label']

                    extra_entry['cls_class'] = none_cls_name
                    extra_entry['label'] = none_cls_index
                    keep2_per_cls.append(extra_entry)
            else:
                if num_Y + num_X > 2 and cls_class == 'Y':
                    group[0]['cls_class'] = none_cls_name
                    group[0]['label'] = none_cls_index
                keep2_per_cls.extend(group)

        # 2 给只有一类的匹配一个长度最相似的
        grouped_data2 = self.__count_by_cls(keep2_per_cls)  # 按照类别统计
        match_one_cls = []

        num_Y = 0
        if 'Y' in grouped_data1.keys():
            num_Y = len(grouped_data1['Y'])

        for cls_class, group in grouped_data2.items():
            if cls_class == none_cls_name: continue
            match_one_cls.extend(group)
            if len(group) == 1:
                if group[0]['cls_class'] == 'Y': continue  # Y染色体只有一条是正常
                if group[0]['cls_class'] == 'X' and num_Y > 0: continue  # 当有Y染色体时，X染色体只有一条是正常
                group_max_length = max(group[0]['crop_patch'].shape)
                if none_cls_name not in grouped_data2.keys(): continue
                closest_value, closest_obj = self.__find_closest_value(grouped_data2[none_cls_name], group_max_length,
                                                                       max_bais_percent)  # 对该组内进行长度匹配
                if closest_value != None:
                    grouped_data2[none_cls_name].remove(closest_obj)
                    closest_obj['cls_class'] = group[0]['cls_class']
                    closest_obj['label'] = group[0]['label']
                    match_one_cls.append(closest_obj)

        if none_cls_name in grouped_data2.keys():
            match_one_cls.extend(grouped_data2[none_cls_name])  # 将剩余未知的加入

        # 3 给没有染色体的组,估计其长度再分配none_cls
        grouped_data3 = self.__count_by_cls(match_one_cls)  # 按照类别统计
        match_zero_cls = []
        for cls_class, group in grouped_data3.items():
            if cls_class == none_cls_name: continue
            match_zero_cls.extend(group)

        no_contain = [none_cls_name, 'X', 'Y']
        zero_cls_length = self.__estimate_length(grouped_data3, no_contain)  # 给没有染色体的类估计一个长度,不包含X,Y和none_cls

        for label, length in zero_cls_length.items():
            for i in range(2):
                if none_cls_name not in grouped_data3.keys(): continue
                closest_value, closest_obj = self.__find_closest_value(grouped_data3[none_cls_name], length,
                                                                       max_bais_percent)  # 对该组内进行长度匹配
                if closest_value != None:
                    grouped_data3[none_cls_name].remove(closest_obj)
                    closest_obj['cls_class'] = self.classes[label]
                    closest_obj['label'] = label
                    match_zero_cls.append(closest_obj)

        if none_cls_name in grouped_data3.keys():
            # match_zero_cls.extend(grouped_data3[none_cls_name])  # 将剩余未知的加入

            # 将未知的还原之前类别
            for extra_entry in grouped_data3[none_cls_name]:
                if 'src_cls_class' in extra_entry.keys() and 'src_label' in extra_entry.keys():
                    extra_entry['cls_class'] = extra_entry['src_cls_class']
                    extra_entry['label'] = extra_entry['src_label']
                    del extra_entry['src_cls_class']
                    del extra_entry['src_label']
                match_zero_cls.append(extra_entry)
        dst = match_zero_cls
        return dst

    def flip(self,src):
        # 图片预处理补白
        img = self.ccp.pad(src, self.model_flip100x.img_max_size)

        infer_results = self.model_flip100x.model.infer(img)

        infer_results = self.model_flip100x.model.postprocess(infer_results)

        # 根据结果对图像翻转
        flip_img = self.model_flip100x.flip(infer_results[0], src)

        is_flip = 0
        if infer_results[0]['pred_class'] == '-':
            is_flip = 1

        return flip_img,is_flip

    def run(self, img_path):

        dst = self.empty

        if not os.path.isfile(img_path):
            return dst

        # 获取文件所在文件夹名作为样本名,并获取文件名称和格式
        img_path = img_path.replace('\\', '/')

        self.logger.info(f'Cls Infer {img_path}')

        start_time = time.time()
        patches = []
        img_floder_path = os.path.dirname(img_path).replace('\\', '/')
        for patch_file in os.listdir(img_floder_path):
            patch_path = os.path.join(img_floder_path, patch_file)
            _, _extension = os.path.splitext(patch_file)
            if not os.path.isfile(patch_path): continue
            if not self.fileOperate.is_type(patch_path, self.file_type): continue
            patch = cv2.imread(patch_path, 1)
            patches.append(patch)

        # 计算该图片所在文件夹内所有图片最大size
        max_size = self.ccp.max_size(patches)
        if max_size > self.img_max_size:
            max_size = self.img_max_size

        src = cv2.imread(img_path, 1)
        img = self.ccp.pad(src, max_size)
        infer_results = self.model.infer(img)
        infer_results = self.model.postprocess(infer_results)

        infer_cls = infer_results[0]['pred_class']
        flip_img, flip_cls = self.flip(src)
        infer_results.append({"is_flip":flip_cls})

        # 保存json文件
        sample, img_file = img_path.split('/')[-2:]
        name, extension = os.path.splitext(img_file)
        save_json_floder = os.path.join(self.save_json_floder, sample)
        self.fileOperate.mkdirs(save_json_floder)
        save_json_path = os.path.join(save_json_floder, f'{name}.json')
        infer_results = self.result_transform(infer_results,img_path)
        self.model.save_json(infer_results, save_json_path)

        # 保存分类图像
        if self.save_vis:
            save_img_floder = os.path.join(self.save_img_floder, sample)
            self.fileOperate.mkdirs(save_img_floder)
            save_img_path = os.path.join(save_img_floder, f'{infer_cls}_{img_file}')
            cv2.imwrite(save_img_path, flip_img)

        end_time = time.time()
        infer_time = round(end_time - start_time, 3)

        self.logger.info(f'Cls {img_path} {infer_cls}, cost {infer_time}s')
        dst = save_json_path.replace('\\', '/')
        return infer_results

    def result_transform(self, results, img_path):
        """
         [
            [
                {
                    "imgPath" : "F:\\adam\\i1000\\test\\singlef\\1-1.jpg_1_1.png"
                },
                {
                    "label" : 6
                },
                {
                    "score" : 0.40368744730949402
                }
            ]
        ]
        :param result:
        :return:
        """
        classes = ['None', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                   '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                   '21', '22', 'X', 'Y']

        image_path = {"image_path": img_path}
        label = {"label": classes.index(results[0]["pred_class"])}
        score = {"score": results[0]["pred_score"]}
        name = {"label_name": results[0]["pred_class"]}
        dst = [[image_path,label,score,name]]

        return dst

class I1000FlipModel(MMInstSegModel):
    def __init__(self, config, logger):

        self.logger = logger
        temp_path = config['TEMP_PATH']
        self.save_json_floder = os.path.join(temp_path, 'json', 'flip').replace('\\', '/')
        self.save_img_floder = os.path.join(temp_path, 'vis', 'flip').replace('\\', '/')

        # 初始化变量
        self.empty = config['EMPTY']
        self.file_type = config['IMAGE_TYPE']
        self.save_vis = config['FLIP100X_SAVE_VIS']
        self.img_max_size = config['FLIP100X_MAXSIZE']

        # 初始化模型
        self.model = MMClsModel(config['FLIP100X_MODEL_CONFIG'], config['FLIP100X_CHECKPOINT'],
                                config['DEVICE'])

        self.ccp = CutCntPatch()
        self.fileOperate = FileOperate()

    def flip(self,result,img):
        if result['pred_class'] == '-':
            img = cv2.flip(img,0)
        return img

    def run(self, img_path):

        dst = self.empty

        # 获取文件所在文件夹名作为样本名,并获取文件名称和格式
        img_path = img_path.replace('\\', '/')
        sample, img_file = img_path.split('/')[-2:]
        name, extension = os.path.splitext(img_file)
        self.logger.info(f'Flip Infer {img_path}')

        start_time = time.time()

        src = cv2.imread(img_path, 1)
        # 图片预处理补白
        img = self.ccp.pad(src, self.img_max_size)

        infer_results = self.model.infer(img)
        infer_results[0]['img_path'] = img_path

        infer_results = self.model.postprocess(infer_results)

        # 根据结果对图像翻转
        flip_img = self.flip(infer_results[0], src)

        # 保存翻转图像
        if self.save_vis:
            save_img_floder = os.path.join(self.save_img_floder, sample)
            self.fileOperate.mkdirs(save_img_floder)
            save_img_path = os.path.join(save_img_floder, img_file)
            cv2.imwrite(save_img_path, flip_img)

        # 保存json文件
        save_json_floder = os.path.join(self.save_json_floder, sample)
        self.fileOperate.mkdirs(save_json_floder)
        save_json_path = os.path.join(save_json_floder, f'{name}.json')
        self.model.save_json(infer_results, save_json_path)
        end_time = time.time()

        infer_time = round(end_time - start_time, 3)
        infer_cls = infer_results[0]['pred_class']
        self.logger.info(f'Flip {img_path} {infer_cls}, cost {infer_time}s')
        dst = save_img_path.replace('\\', '/')
        return dst

class I1000DetModel(MMDetModel):
    def __init__(self, config, logger):

        self.logger = logger
        temp_path = config['TEMP_PATH']
        self.save_img_floder = os.path.join(temp_path, 'vis', 'det').replace('\\', '/')
        self.save_json_floder = os.path.join(temp_path, 'json', 'det').replace('\\', '/')

        # 初始化变量
        self.empty = config['EMPTY']
        self.file_type = config['IMAGE_TYPE']

        self.save_vis = config['DET10X_SAVE_VIS']
        self.conf_score = config['DET10X_CONFIDENCE_SCORE']
        self.iou_score = config['DET10X_IOU_THRESH']

        # 初始化模型
        self.model = MMDetModel(config['DET10X_MODEL_CONFIG'], config['DET10X_CHECKPOINT'],
                                    config['DEVICE'])

        self.fileOperate = FileOperate()
    def run(self, img_path):
        img_path = img_path.replace('\\', '/')
        self.logger.info(f'Det Infer {img_path}')

        start_time = time.time()

        infer_result = self.model.infer(img_path)

        infer_result = self.model.postprocess(infer_result, self.conf_score)
        # NMS去重
        infer_result = self.model.nms(infer_result, self.iou_score)

        infer_result = self.postprocess_score(infer_result)

        if self.save_vis:
            self.fileOperate.mkdirs(self.save_img_floder,False)
            self.model.visual(img_path, infer_result, self.save_img_floder)

        img_file = os.path.basename(img_path)
        name, extension = os.path.splitext(img_file)
        save_json_path = os.path.join(self.save_json_floder, f'{name}.json').replace('\\', '/')
        infer_result = self.result_transform(infer_result,img_path)
        self.model.save_json(infer_result, save_json_path)
        self.logger.info(f'Save Json  {save_json_path}')

        end_time = time.time()
        infer_time = round(end_time - start_time, 3)
        self.logger.info(f'Det successufl {len(infer_result)-1} objs, cost {infer_time}s')

        return infer_result
    def postprocess_score(self,results):
        for idx,result in enumerate(results):
            label = result["label"]
            if label == 0:
                result["score"] += 4
            elif label == 1:
                result["score"] += 3
            elif label == 2:
                result["score"] += 2
            else:
                result["score"] += 1

        return results
    def result_transform(self,results,img_path):
        """
        [
            {
                "image_path" : "F:/adam/i1000/test/10x\\10.jpg"
            },
            {
                "box" :
                [
                    1670,
                    1111,
                    1739,
                    1185
                ],
                "id" : 0,
                "label" : 3,
                "score" : 1.5031586885452271
            },
            ...
        ]
        :param result:
        :return:
        """

        image_path = {"image_path":img_path}
        dst = [image_path]
        for idx,result in enumerate(results):
            dst.append({
                "box": result["bbox"],
                "id" : idx,
                "label": result["label"],
                "score": result["score"],

            })
        return dst

class I1000MaskModel(MMSegModel):
    def __init__(self, config, logger):

        self.logger = logger
        temp_path = config['TEMP_PATH']
        self.save_img_floder = os.path.join(temp_path, 'vis', 'mask').replace('\\', '/')

        # 初始化变量
        self.empty = config['EMPTY']
        self.file_type = config['IMAGE_TYPE']
        self.save_vis = config['MASK100X_SAVE_VIS']

        # 初始化模型
        self.model = MMSegModel(config['MASK100X_MODEL_CONFIG'], config['MASK100X_CHECKPOINT'],
                                config['DEVICE'])

        self.basicCVAlg = BasicCVAlg()
        self.fileOperate = FileOperate()

    def run(self, img_path):
        img_path = img_path.replace('\\', '/')
        self.logger.info(f'Mask Infer {img_path}')

        start_time = time.time()

        # 分割染色体
        infer_result = self.model.infer(img_path)
        masks = self.model.get_masks(infer_result)

        if self.save_vis:
            # self.model.save_masks(infer_result,self.save_img_floder)
            self.model.visual(infer_result, self.save_img_floder)

        end_time = time.time()
        infer_time = round(end_time - start_time, 3)
        self.logger.info(f'Mask successufl, cost {infer_time}s')

        return masks

class I1000AnalyseModel():
    def __init__(self, config, logger,models_dict):
        self.logger = logger
        self.empty = config['EMPTY']
        self.cls_model = config['CLS_MODEL']

        # 初始化模型
        self.model_cls100x = models_dict['100XCls']# model_cls100x
        self.model_flip100x = models_dict['100XFlip']
        self.mask_seg = config['MASKSEG']
        self.model_seg100x = models_dict['100XInstSeg']


        if self.mask_seg:
            self.model_mask100x = models_dict['100XMask']

        self.fileOperate = FileOperate()

    def __cls_flip_single_chroms(self, img_path, results):
        """
        对单个染色体图像进行类别和是否需要旋转倒置预测，根据结果调整和保存单条染色体图像。

        参数：
        img_path (str): 输入的染色体图像文件路径。
        results (list): 包含边界等信息的结果列表。

        返回：
        results (list): 经过类别和比例预测后的结果列表。
        """
        img = cv2.imread(img_path, 1)
        crop_patches = []

        img_file = os.path.basename(img_path)
        name, extension = os.path.splitext(img_file)
        save_patch_path = os.path.join(self.model_seg100x.save_img_floder, name, 'single')

        if not os.path.exists(save_patch_path):
            os.makedirs(save_patch_path)

        for idx, result in enumerate(results):
            cnt = result['cnts']
            patch = self.model_cls100x.ccp.crop(img, cnt, isRotate=True)
            crop_patches.append(patch)

        max_size = self.model_cls100x.ccp.max_size(crop_patches)

        # 获取类别
        for idx, crop_patch in enumerate(crop_patches):
            if self.cls_model:
                cls_pad_patch = self.model_cls100x.ccp.pad(crop_patch, max_size)
                cls_result = self.model_cls100x.model.infer(cls_pad_patch)[0]
                score = round(cls_result['pred_score'], 3)
                results[idx]['label'] = cls_result['pred_label']
                results[idx]['cls_class'] = cls_result['pred_class']
                results[idx]['cls_score'] = score
                results[idx]['crop_patch'] = crop_patch
            else:
                results[idx]['cls_class'] = self.model_cls100x.classes[results[idx]['label']]
                results[idx]['cls_score'] = results[idx]['score']
                results[idx]['crop_patch'] = crop_patch

        if self.cls_model:
            # 不符合规则的匹配类别
            results = self.model_cls100x.modify_cls(results)

        # 旋转倒置
        for idx, result in enumerate(results):
            crop_patch = result['crop_patch']
            flip_pad_patch = self.model_cls100x.ccp.pad(crop_patch, self.model_flip100x.img_max_size)
            flip_result = self.model_flip100x.model.infer(flip_pad_patch)[0]
            score = results[idx]['cls_score']
            cls = results[idx]['cls_class']
            if self.model_seg100x.save_vis:
                save_path = os.path.join(save_patch_path, f'{cls}_{score}_{idx}.png')
                flip_patch = self.model_flip100x.flip(flip_result, crop_patch)
                cv2.imwrite(save_path, flip_patch)

        return results,save_patch_path

    def preprocess(self,img_path):

        image = cv2.imread(img_path, 1)
        if self.mask_seg:
            masks = self.model_mask100x.run(img_path)
            mask = masks['chromosome']
            image[mask == 0, :] =  255

        return image

    def run(self, img_path):
        img_path = img_path.replace('\\', '/')
        dst = self.empty
        self.logger.info(f'Analyse {img_path}')

        img_file = os.path.basename(img_path)
        name, extension = os.path.splitext(img_file)
        save_patch_path = os.path.join(self.model_seg100x.save_img_floder, name)
        self.fileOperate.del_dir(save_patch_path)

        start_time = time.time()

        # 分割染色体核型
        image = self.preprocess(img_path)
        seg_result = self.model_seg100x.run(img_path, image)

        if len(seg_result) == 0:
            return dst

        # 分类和旋转单条染色体
        infer_result,save_single_path = self.__cls_flip_single_chroms(img_path, seg_result)

        infer_result = self.model_seg100x.del_key(infer_result, 'crop_patch')

        self.fileOperate.mkdirs(self.model_seg100x.save_json_floder)
        save_json_path = os.path.join(self.model_seg100x.save_json_floder, f'{name}.json').replace('\\', '/')
        self.logger.info(f'Save Json  {save_json_path}')

        infer_result = self.result_transform(infer_result, img_path,save_single_path)
        self.model_seg100x.save_json(infer_result, save_json_path)

        end_time = time.time()
        infer_time = round(end_time - start_time, 3)
        self.logger.info(f'Analyse successufl cost {infer_time}s')
        dst = save_json_path.replace('\\', '/')
        return infer_result

    def result_transform(self,results,img_path,save_single_path):
        """
        [
            {
                "image_path" : "F:/adam/i1000/test/100x\\1.png"
            },
            {
                "single_path" : "F:/adam/i1000/test/100x"
            },
            {
                "class" : 3,
                "number" : 1,
                "points" :
                [
                    [
                        466,
                        1183
                    ],
                    ...
                ]
            },
            ...
            {
                "score" : 1,
                "scorevalue" : 44
            }
        ]
        :param result:
        :return:
        """
        classes = ['None', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                   '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                   '21', '22', 'X', 'Y']

        image_path = {"image_path":img_path}
        seg_floder= {"seg_floder":save_single_path}
        dst = [image_path,seg_floder]
        for idx,result in enumerate(results):
            dst.append({
                "class": classes.index(result["cls_class"]),
                "class_name": result["cls_class"],
                "class_score": result["cls_score"],
                "seg_score": result["score"],
                "number": idx+1,
                "points":result["cnts"][0]

            })
        score_cls,score_value = self.model_seg100x.chrom_score(result)
        dst.append({"score" : score_cls,"scorevalue" : score_value})
        return dst

class I1000AnalyseLabelmeJsons():
    def __init__(self,config, logger):
        self.logger = logger
        self.jsons_path = config['JSONS_PATH']
        self.empty = config['EMPTY']
        self.classes = ['None', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                        '21', '22', 'X', 'Y']
        temp_path = config['TEMP_PATH']
        self.save_img_floder = os.path.join(temp_path, 'vis', 'seg').replace('\\', '/')
        self.save_json_floder = os.path.join(temp_path, 'json', 'seg').replace('\\', '/')

    def save_json(self,results,save_path):
        print(f'Save {save_path}')
        directory_path = os.path.dirname(save_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        if save_path  != None:
            with open(save_path, 'w') as json_file:
                json.dump(results, json_file, indent=4)

    def __analyse_image_path(self,image_path):
        image_path = image_path.replace('\\','/')
        self.logger.info(f'Analyse {image_path}')
        image_name = image_path.split('/')[-2]
        name,_ = os.path.splitext(image_name)

        json_name = f'{name}.json'
        json_path = os.path.join(self.jsons_path,json_name)
        return json_path,name

    def __get_cnts(self,json_path):

        json_file = open(json_path, 'r')
        json_dict = json.load(json_file)

        if 'shapes' not in json_dict.keys() or len(json_dict['shapes']) <= 0:
            json_file.close()
            return self.empty

        shapes = json_dict['shapes']
        cnts = []
        for shape in shapes:
            cnt = shape['points']
            cnts.append(cnt)

        return cnts

    def __sort_cnts_by_area(self,cnts):
        area_cnts = []
        for idx,cnt in enumerate(cnts):
            if len(cnt) < 5: continue
            cnt_array = np.array(cnt,dtype=np.float32)
            area = cv2.contourArea(cnt_array)
            area_cnts.append([area,cnt])

        area_cnts = sorted(area_cnts, key=lambda x: x[0], reverse=True)

        dst = []
        for area, cnt in area_cnts:
            dst.append(cnt)

        return dst

    def __get_cnts_cls_by_area(self,cnts):
        cnts = self.__sort_cnts_by_area(cnts)
        dst = []
        for idx, cnt in enumerate(cnts):
            cnt_cls = {}
            cls_idx = 0
            if idx < 46:
                cls_idx = int(idx / 2) + 1

            cls_class = self.classes[cls_idx]
            cnt_cls['cls_class'] = cls_class
            cnt_cls['cnt'] = cnt
            dst.append(cnt_cls)
        return dst

    def result_transform(self,cnts,dst):
        """
        [
            {
                "image_path" : "F:/adam/i1000/test/100x\\1.png"
            },
            {
                "single_path" : "F:/adam/i1000/test/100x"
            },
            {
                "class" : 3,
                "number" : 1,
                "points" :
                [
                    [
                        466,
                        1183
                    ],
                    ...
                ]
            },
            ...
            {
                "score" : 1,
                "scorevalue" : 44
            }
        ]
        :param result:
        :return:
        """

        cls_cnts = self.__get_cnts_cls_by_area(cnts)
        self.logger.info(f"Analyse {len(cls_cnts)} objects")

        for idx,cls_cnt in enumerate(cls_cnts):
            dst.append({
                "class": self.classes.index(cls_cnt["cls_class"]),
                "class_name": cls_cnt["cls_class"],
                "class_score": 1,
                "seg_score": 1,
                "number": idx+1,
                "points":cls_cnt['cnt']

            })

        dst.append({"score" : 1,"scorevalue" : 46})
        return dst

    def run(self,image_path):

        dst = self.empty
        try:
            json_path,name = self.__analyse_image_path(image_path)
            self.logger.info(f'Analyse Json {json_path}')
            if not os.path.exists(json_path):
                self.logger.info(f"Don't find {json_path}")
                return dst

            cnts = self.__get_cnts(json_path)
            image_path = {"image_path": image_path}
            seg_floder = {"seg_floder": ''}
            dst = [image_path, seg_floder]

            dst = self.result_transform(cnts,dst)
            save_json_path = os.path.join(self.save_json_floder,f'{name}.json')
            self.save_json(dst,save_json_path)

        except Exception as e:
            self.logger.info(f"Error: {e} in analyse: {json_path} ")
            dst = self.empty

        return dst
