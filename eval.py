from vocab import Vocabulary
import evaluation_models
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
evaluation_models.evalrank("runs/model_vsrn_coco_1.pth.tar", "runs/model_vsrn_coco_2.pth.tar",data_path="data", split="testall", fold5=True)
