from vocab import Vocabulary
import evaluationAOQ
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
evaluationAOQ.evalrank("runs/model_bfan_f30k_equal.pth.tar", "runs/model_bfan_f30k_prob.pth.tar",data_path="data", split="test", fold5=False)
#evaluationAOQ.evalrank("runs/model_bfan_coco_equal.pth.tar", "runs/model_bfan_coco_prob.pth.tar",data_path="data", split="testall", fold5=True)
