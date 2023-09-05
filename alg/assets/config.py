VERSION = "01.01.15-Beta1"
HOST = "127.0.0.1"
PORT1 = 8886
PORT2 = 8887
CLS_MODEL = True

TEMP_PATH = ''

if CLS_MODEL:
    SEG100X_MODEL_CONFIG = 'alg/model/config/I1000SEG_queryinst_r50.py'
    SEG100X_CHECKPOINT = 'alg/model/checkpoint/I1000SEG_queryinst_r50.pth'
else:
    SEG100X_MODEL_CONFIG = 'alg/model/config/I1000SEGCLS_queryinst_r50.py'
    SEG100X_CHECKPOINT = 'alg/model/checkpoint/I1000SEGCLS_queryinst_r50.pth'

SEG100X_IOU_THRESH = 0.3
SEG100X_CONFIDENCE_SCORE = 0.1
SEGCV_POST = False

MASKSEG = True
if MASKSEG:
    MASK100X_MODEL_CONFIG = 'alg/model/config/I1000SEG_unet.py'
    MASK100X_CHECKPOINT = 'alg/model/checkpoint/I1000SEG_unet.pth'
    MASK100X_SAVE_VIS = True

DET10X_MODEL_CONFIG = 'alg/model/config/I1000DET_yolox_s.py'
DET10X_CHECKPOINT = 'alg/model/checkpoint/I1000DET_yolox_s.pth'
DET10X_IOU_THRESH = 0.3
DET10X_CONFIDENCE_SCORE = 0.2


CLS100X_MODEL_CONFIG = 'alg/model/config/I1000CLS_resnet_50.py'
CLS100X_CHECKPOINT = 'alg/model/checkpoint/I1000CLS_resnet_50.pth'
CLS100X_MAXSIZE = 512

FLIP100X_MODEL_CONFIG = 'alg/model/config/I1000FLIP_resnet_50.py'
FLIP100X_CHECKPOINT = 'alg/model/checkpoint/I1000FLIP_resnet_50.pth'
FLIP100X_MAXSIZE = 256

CLS100X_SAVE_VIS = True
SEG100X_SAVE_VIS = True
DET10X_SAVE_VIS = True
FLIP100X_SAVE_VIS = True
DEVICE = 'cuda:0'

DEBUG = False
TEST_INIT = True
EMPTY = ' '
IMAGE_TYPE = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPEG', '.JPG', '.PNG', '.BMP', '.TIF', '.TIFF']
# JSONS_PATH = 'F:/adam/i1000/seg_100x/src/jsons'
JSONS_PATH = ''