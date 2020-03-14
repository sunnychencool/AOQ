# -----------------------------------------------------------
# A Bidirectional Focal Attention Network implementation based on 
# https://arxiv.org/abs/1909.11416.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Chunxiao Liu, 2019
# ---------------------------------------------------------------
"""Data provider"""

import torch
import torch.utils.data as data
import os
import numpy as np
import random

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab, offcandipath=None, offcandisize=None):
        self.vocab = vocab
        loc = data_path + '/'

        # Captions (word ids)
        self.captions = []
        with open(loc+'%s_precaps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000
        self.split = data_split
        if data_split == 'train':
           i2tpath = offcandipath+'i2t_'+data_path.split('/')[-1].split('_')[0]+'.npy'
           t2ipath = offcandipath+'t2i_'+data_path.split('/')[-1].split('_')[0]+'.npy'
           self.i2thn = np.load(i2tpath)
           self.t2ihn = np.load(t2ipath)         
           self.offcandisize = offcandisize
    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index/self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # Word ids.
        caps = []
        caps.extend(caption.split(','))
        caps = map(int, caps)

        target = torch.Tensor(caps)

        if self.split == 'train':
          
          # load candi index list 
          captionhncandi = self.i2thn[img_id]
          # randomly sampled the nth top candidate, get the index
          n = random.randint(0,self.offcandisize-1)
          capindex = captionhncandi[n]
          # get offline hard negative caption
          captionhn = self.captions[int(capindex)]
          capshn = []
          capshn.extend(captionhn.split(','))
          capshn = map(int, capshn)
          targethn = torch.Tensor(capshn)
          
          imagehncandi = self.t2ihn[index]
          # noted that for both datasets the training caption number is 5 times larger than the training image number
          n = n/5
          imgindex = imagehncandi[n]
          # get offline hard negative image
          imagehn = torch.Tensor(self.images[int(imgindex)])

           # get the positive caption of the offline hard negative image
          n2 = random.randint(0,4)
          capindexhn2 = imgindex * 5 + n2
          captionhn2 = self.captions[int(capindexhn2)]
          
          capshn2 = []
          capshn2.extend(captionhn2.split(','))
          capshn2 = map(int, capshn2)
          targethn2 = torch.Tensor(capshn2)
          # get the positive image of the offline hard negative caption
          imgindexhn2 = capindex/5
          imagehn2 = torch.Tensor(self.images[int(imgindexhn2)])

          #imagehn = torch.Tensor(self.images[0])
        else:
          # just define these variables without using it for validation/test
          imagehn, targethn = image, target
          imagehn2, targethn2 = image, target
        return image, target, index, img_id, imagehn, targethn, imagehn2, targethn2

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, imageshn, captionshn, imageshn2, captionshn2 = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    imageshn = torch.stack(imageshn, 0)
    imageshn2 = torch.stack(imageshn2, 0)

    #caption_labels_ = torch.stack(caption_labels, 0)
    #caption_masks_ = torch.stack(caption_masks, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengthshn = [len(cap) for cap in captionshn]
    targetshn = torch.zeros(len(captionshn), max(lengthshn)).long()
    for i, cap in enumerate(captionshn):
        end = lengthshn[i]
        targetshn[i, :end] = cap[:end]

    lengthshn2 = [len(cap) for cap in captionshn2]
    targetshn2 = torch.zeros(len(captionshn2), max(lengthshn2)).long()
    for i, cap in enumerate(captionshn2):
        end = lengthshn2[i]
        targetshn2[i, :end] = cap[:end]

    
    return images, targets, lengths, ids, imageshn, targetshn, lengthshn, imageshn2, targetshn2, lengthshn2


def get_precomp_loader(data_path, data_split, vocab, opt, offcandipath=None, offcandisize=None, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if data_split == 'train':
        dset = PrecompDataset(data_path, data_split, vocab, offcandipath, offcandisize)  
    else:
        dset = PrecompDataset(data_path, data_split, vocab) 

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, offcandipath, offcandisize, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt, offcandipath, offcandisize, 
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader
