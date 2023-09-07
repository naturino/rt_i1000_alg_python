import os.path

import torch
import threading
from mmengine.fileio import dump

from utils import loger, socket_server
from i1000_alg import *


class DeepCell():
    def __init__(self, config_path):
        root_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
        # 读取配置文件
        self.config = self._read_config(config_path)
        self.config['ROOT_PATH'] = root_path
        # 初始化配置文件
        config_json = dump(self.config, file_format='json', indent=4)
        temp_path = self.config['TEMP_PATH']
        if self.config['TEMP_PATH'] == '':
            self.config['TEMP_PATH'] = os.path.join(self.config['ROOT_PATH'], 'alg', 'temp').replace('\\', '/')
            temp_path = self.config['TEMP_PATH']

        self.file_operate = FileOperate()
        self.file_operate.del_dir(temp_path)

        # 初始化日志
        self.logger = loger.Logger(os.path.join(self.config['ROOT_PATH'], 'alg', 'log'))
        self.logger.info(f"Config Info: {config_json}")
        self.logger.info(f"DeepCell Start")

        # 初始化变量
        self.empty = self.config['EMPTY']
        self.file_type = self.config['IMAGE_TYPE']

        # 创建一个锁对象
        self.run_lock = threading.Lock()

        # 启动模型
        model_state = self._start_model(self.config)

        if model_state:
            # 开启通信
            self._start_socket(self.config)

    def run(self, info):
        with self.run_lock:
            dst = self.empty
            task, img_path = self._get_task(info)

            if not os.path.isfile(img_path):
                self.logger.info(f'{img_path} is not file')
                return dst
            if not self.file_operate.is_type(img_path, self.file_type):
                self.logger.info(f'{img_path} is not in {self.file_type}')
                return dst
            if task not in self.task_model.keys():
                self.logger.info(f"Dont't find {task} model")
                return dst

            dst = self.task_model[task].run(img_path)

            if type(dst) != str:
                dst = json.dumps(dst, indent=4)

            return dst

    def _start_model(self,config):

        self.task_model = {}

        self.jsons_path = config['JSONS_PATH']
        self.only_analyse_json = False

        if self.jsons_path != '':
            self.only_analyse_json = True

        gpu_is_available = False
        gpu_info = self._get_gpu_info()
        self.logger.info(f"Find GPU: {gpu_info}")
        if gpu_info == 'NVIDIA GeForce RTX 3060':
            gpu_is_available = True
        is_cpu = config['DEVICE'] == 'cpu'

        self.task_model['100XJson'] = I1000AnalyseLabelmeJsons(config, self.logger)

        if gpu_is_available or is_cpu:
            # 初始化算法模型
            self.task_model['100XFlip'] = I1000FlipModel(config, self.logger)
            self.task_model['100XCls'] = I1000ClsModel(config, self.logger, self.task_model['100XFlip'])


            self.task_model['10XDet'] = I1000DetModel(config, self.logger)
            self.task_model['100XInstSeg'] = I1000InstSegModel(config, self.logger)

            if config['MASKSEG']:
                self.task_model['100XMask'] = I1000MaskModel(config, self.logger)

            self.task_model['100XAnalyse'] = I1000AnalyseModel(config, self.logger, self.task_model)

            self.logger.info(f"Model Load Successful")

        else:
            self.logger.info(f"Don't find device...")
            return False

        # 测试模型
        is_test = config['TEST_INIT']
        if is_test:
            self._model_test()

        debug = config['DEBUG']
        if debug:
            self._model_debug()

        return True

    def _start_socket(self,config):
        self.host = config['HOST']
        self.port1 = config['PORT1']
        self.port2 = config['PORT2']

        # 启动socket1
        self.server1 = socket_server.SocketServer(self.host, self.port1, self.logger)
        self.logger.info(f"SocketServer Init {self.host}:{self.port1} Successful")

        # 启动socket2
        self.server2 = socket_server.SocketServer(self.host, self.port2, self.logger)
        self.logger.info(f"SocketServer Init {self.host}:{self.port2} Successful")

        # 创建两个线程来分别启动这两个服务
        thread1 = threading.Thread(target=self.server1.start, args=(self.run,))
        thread2 = threading.Thread(target=self.server2.start, args=(self.run,))

        # 启动线程
        thread1.start()
        thread2.start()

        try:
            # 在主线程中等待两个线程完成
            thread1.join()
            thread2.join()
        except KeyboardInterrupt:

            self.server1.stop()
            self.server2.stop()

    def _read_config(self, path):
        dst = {}
        exec(open(path).read(), dst)
        del_key = '__builtins__'
        if del_key in dst.keys():
            del dst[del_key]
        return dst

    def _get_gpu_info(self):
        gpu_info = ''
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            gpu_info = gpu.name
        return gpu_info

    def _model_test(self):
        self.run(os.path.join(self.config['ROOT_PATH'],'alg/assets/img/demo1/1_7.png*0'))
        self.run(os.path.join(self.config['ROOT_PATH'],'alg/assets/img/demo1/1_28.png*1'))
        self.run(os.path.join(self.config['ROOT_PATH'],'alg/assets/img/demo3.png*2'))
        self.run(os.path.join(self.config['ROOT_PATH'],'alg/assets/img/demo2.jpg'))
        self.logger.info(f"Model Test Successful")

    def _model_debug(self):
        floder = 'F:/adam/i1000/adk/adk_src'
        for root, _, file_list in os.walk(floder):
            for file in file_list:
                path = os.path.join(root, file) + '*2'
                self.run(path)

    def _get_task(self,info):
        dst = ''
        info_list = info.split('*')
        image_path = info_list[0]

        if len(info_list) == 2:
            img_path, task = info_list
            if task == '0':
                dst = '100XFlip'

            elif task == '1':
                dst = '100XCls'

            elif task == '2':
                if self.only_analyse_json:
                    dst = '100XJson'
                else:
                    dst = '100XAnalyse'

            # elif task == '3':
            #     dst = '100XMask'

        elif len(info_list) == 1:
            dst = '10XDet'

        return dst, image_path

if __name__ == "__main__":
    deepcell_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(deepcell_path, 'alg/assets/config.py')
    DeepCell(config_path)
