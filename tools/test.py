import os
import cv2
import numpy as np
import json
import base64

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            png_bytes = cv2.imencode('.png', obj)[1].tobytes()
            # 使用Base64编码将字节对象转换为字符串
            base64_string = base64.b64encode(png_bytes).decode('utf-8')
            return base64_string
        else:
            return super(MyEncoder, self).default(obj)

def copy_mask_file(path,save):
    if not os.path.exists(save):
        os.makedirs(save)
    for root,_,files in os.walk(path):
        for file in files:
            if 'chromosome' in file:
                path = os.path.join(root,file)
                src = cv2.imread(path)
                # shape = src.shape
                # bytsdata = data_trans.mat2bytes(src)
                # dst = bytsdata
                # src = [{'src':src}]
                # dst = json.dumps(src, indent=4,cls=MyEncoder)
                # 编码为PNG格式的字节对象
                png_bytes = cv2.imencode('.png', src)[1].tobytes()

                # 使用Base64编码将字节对象转换为字符串
                base64_string = base64.b64encode(png_bytes).decode('utf-8')

                # 打印Base64编码后的字符串
                print(base64_string)

                # 将Base64编码的字符串解码回二进制数据
                decoded_bytes = base64.b64decode(base64_string)

                # 将解码后的数据重新转换为图像
                decoded_image = cv2.imdecode(np.frombuffer(decoded_bytes, np.uint8), cv2.IMREAD_COLOR)

                save_name = root.split('\\')[-1]
                # dst = os.path.join(save,save_name + '.png')
                # shutil.copy2(src,dst)
                # print(dst)

if __name__ == '__main__':
    src= 'E:/adam/code/python/i1000/DeepCell/alg/temp/vis/mask'
    dst= 'E:/adam/code/python/i1000/DeepCell/alg/temp/vis/dst'
    ehance = 'E:/adam/code/python/i1000/DeepCell/alg/temp/vis/ehance'
    copy_mask_file(src, dst)
    if not os.path.exists(ehance):
        os.makedirs(ehance)