import pickle
import os
import time
import shutil

import torch
import os 
import numpy as np
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
totalsavecandi = 2000
inf = 9999
def main():
    # img_embstrain: (n,2048) image features predicted by first-round VSRN
    # caps_embstrain: (5n,2048) caption features predicted by first-round VSRN    
    img_embs = np.load('../data/img_embstrain.npy')
    cap_embs = np.load('../data/cap_embstrain.npy')
    print(np.shape(img_embs), np.shape(cap_embs))
    img_embs = torch.from_numpy(img_embs).half().cuda()
    cap_embs = torch.from_numpy(cap_embs).half().cuda()    
    i2thn = np.zeros((len(img_embs),totalsavecandi)).astype('int32')
    for i in range(len(img_embs)):
        im = img_embs[i:i+1]
        simi = (im * cap_embs).sum(1)
        if i%50==0:
            scoret, topt = torch.sort(simi,descending=True)
            topt = topt.cpu().numpy().copy()
            print(i, np.where(topt==(i*5)), np.where(topt==(i*5+1)), np.where(topt==(i*5+2)), np.where(topt==(i*5+3)), np.where(topt==(i*5+4))) 
        simi[i*5:i*5+5] = -inf
        score, top = torch.topk(simi,totalsavecandi)
        i2thn[i] = top.cpu().numpy().copy().astype('int32')
    np.save('../offlinecandidates/i2t_coco.npy', i2thn)


if __name__ == '__main__':
    main()
