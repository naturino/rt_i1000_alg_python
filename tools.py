import cv2
import os
import shutil
import cut_cnt_patch
import file_operate

class SegByReport:
    def __init__(self):
        self.cpp = cut_cnt_patch.CutCntPatch()
        self.fp = file_operate.FileOperate()
        self.cls_num_row = [5,7,6,5]
        self.cls = ['1','2','3','4','5','6','7','8','9','10',
                  '11','12','13','14','15','16','17','18','19','20',
                  '21','22','X','Y']

    def getCntsCls(self,cnts):

        cnts_xy = []
        # 按照Y的坐标给轮廓排序
        for idx,cnt in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(cnt)
            cnts_xy.append({'cnt': cnt,'x':x, 'y':y+h})
        y_sorted = sorted(cnts_xy, key=lambda item: item['y'],reverse=False)

        y_idx = 0
        cls = 0
        dst = []
        total_num = 0
        for row_num in self.cls_num_row:
            num = row_num * 2
            # 根据规则数量选出每一行
            row_cnts = y_sorted[y_idx:y_idx+num]
            # 按照X的坐标给该行轮廓排序
            x_sorted = sorted(row_cnts, key=lambda item: item['x'])
            y_idx += num

            # 按顺序指定类别
            for x_idx,x_cnt in enumerate(x_sorted):
                dst_dict = {}
                dst_dict['cnt'] = x_cnt['cnt']
                dst_dict['cls'] = self.cls[cls]

                # 最后一条染色体长度小于前一条的80%则为Y染色体
                if total_num == 45:
                    _,_,_,h1 = cv2.boundingRect(dst[-1]['cnt'])
                    _, _, _, h2 = cv2.boundingRect(x_cnt['cnt'])
                    if h2 < h1 * 0.8:
                        dst_dict['cls'] = self.cls[cls+1]
                dst.append(dst_dict)
                if total_num % 2==1:
                    cls +=1
                    # print(cls)
                total_num +=1

        return dst

    def getCntsPatch(self,img,cnts):
        dst = []
        for idx,cnt_dict in enumerate(cnts):
            cnt = cnt_dict['cnt']
            patch = self.cpp.crop(img,cnt)
            cnt_dict['patch'] = patch
            dst.append(cnt_dict)
        return dst

    def padPatches(self,patches_dict_list):
        patches_list = [patches_dict['patch'] for patches_dict in patches_dict_list]
        img_max_size = self.cpp.max_size(patches_list)
        for patches_dict in patches_dict_list:
            patch = patches_dict['patch']
            patch = self.cpp.pad(patch, img_max_size, color=255)
            patches_dict['patch'] = patch
        return patches_dict_list

    def savePatch(self,patches_dict_list,save_floder):
        self.fp.mkdirs(save_floder)
        for idx,patches_dict in enumerate(patches_dict_list):
            patch = patches_dict['patch']
            cls = patches_dict['cls']
            path = os.path.join(save_floder,f'{cls}_{idx}.png')
            cv2.imwrite(path,patch)
            # print(f'Save {path}')

    def fliterCnts(self,cnts,area_thresh):
        dst = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)

            if area < area_thresh:
                continue

            if max(w,h) > 300:
                continue
            dst.append(cnt)
        return dst

    def run(self,img_path,save):

        img = cv2.imread(img_path, 1)
        floder = file.split('.')[0]

        save_floder = os.path.join(save,floder)

        cnts = self.cpp.getImgCnts(img,False)
        cnts = self.fliterCnts(cnts, 400)

        if not os.path.exists(save_floder):
            os.makedirs(save_floder)

        if len(cnts) != 46:
            basename = os.path.basename(img_path)
            print(basename,img_path)
            shutil.copy2(img_path,os.path.join(save,basename))
            return


        cls_dict_list = self.getCntsCls(cnts)

        patches_dict_list = self.getCntsPatch(img, cls_dict_list)
        # patches_dict_list = self.padPatches(patches_dict_list)
        self.savePatch(patches_dict_list,save_floder)

if __name__ == '__main__':
    my_seg = SegByReport()
    img_floder = "F:/adam/i1000/adk/adk_seg"
    save = "F:/adam/i1000/adk/adk_report"
    for file in os.listdir(img_floder):
        img_path = os.path.join(img_floder, file)
        my_seg.run(img_path,save)

