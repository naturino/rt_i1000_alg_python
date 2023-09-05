import os
import json
from mmpretrain.apis import ImageClassificationInferencer as mmpre_infer

class MMClsModel:

    def __init__(self,config_file,checkpoint_file,device):
        self.model = mmpre_infer(config_file, pretrained=checkpoint_file, device=device)

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
