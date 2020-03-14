# -------------------------------------------------------------------------------------
# A Bidirectional Focal Atention Network implementation based on
# https://arxiv.org/abs/1909.11416.
# "Focus Your Atention: A Bidirectional Focal Atention Network for Image-Text Matching"
# Chunxiao Liu, Zhendong Mao, An-An Liu, Tianzhu Zhang, Bin Wang, Yongdong Zhang
#
# Writen by Chunxiao Liu, 2019
# -------------------------------------------------------------------------------------
"""BFAN model"""

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError(
            "Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers,
                          batch_first=True, bidirectional=True)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        sorted_input_lengths_list = np.sort(lengths)[::-1].tolist() # list of sorted input_lengths
        sort_ixs = np.argsort(lengths)[::-1].tolist() # list of int sort_ixs, descending
        s2r = {s: r for r, s in enumerate(sort_ixs)} # O(n)
        recover_ixs = [s2r[s] for s in range(len(lengths))]  # list of int recover ixs


            # move to long tensor
        sort_ixs = x.data.new(sort_ixs).long()  # Variable long
        recover_ixs = x.data.new(recover_ixs).long()
        
        x = x[sort_ixs]
        x = self.embed(x)
        packed = pack_padded_sequence(x, sorted_input_lengths_list, batch_first=True)


        # Forward propagate RNN
        out, _ = self.rnn(packed)


        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb = padded[0][recover_ixs]
        cap_len = padded[1][recover_ixs]

        #I = torch.LongTensor(lengths).view(-1, 1, 1)
        #I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        #out = torch.gather(out, 1, I).squeeze(1)

        #if self.use_bi_gru:
        cap_emb = (cap_emb[:,:,:cap_emb.size(2)/2] + cap_emb[:,:,cap_emb.size(2)/2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len


def func_attention(query, context, opt, eps=1e-8):
    """
    query: (batch, queryL, d)
    context: (batch, sourceL, d)
    opt: parameters
    """
    batch_size, queryL, sourceL = context.size(
        0), query.size(1), context.size(1)

    # Step 1: preassign attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    attn = torch.bmm(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn*opt.lambda_softmax)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    # Step 2: identify irrelevant fragments
    # Learning an indicator function H, one for relevant, zero for irrelevant
    if opt.focal_type == 'equal':
        funcH = focal_equal(attn, batch_size, queryL, sourceL)
    elif opt.focal_type == 'prob':
        funcH = focal_prob(attn, batch_size, queryL, sourceL)
    else:
        raise ValueError("unknown focal attention type:", opt.focal_type)

    # Step 3: reassign attention
    tmp_attn = funcH * attn
    attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
    re_attn = tmp_attn / attn_sum

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # --> (batch, sourceL, queryL)
    re_attnT = torch.transpose(re_attn, 1, 2).contiguous()
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, re_attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext


def focal_equal(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as equal
    sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
    attn: (batch, queryL, sourceL)
    """
    funcF = attn * sourceL - torch.sum(attn, dim=-1, keepdim=True)
    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn


def focal_prob(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as the sqrt 
    of their similarity probability to the query fragment
    sigma_{j} (xi - xj)gj = sigma_{j} xi*gj - sigma_{j} xj*gj
    attn: (batch, queryL, sourceL)
    """

    # -> (batch, queryL, sourceL, 1)
    xi = attn.unsqueeze(-1).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj = attn.unsqueeze(2).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj_confi = torch.sqrt(xj)

    xi = xi.view(batch_size*queryL, sourceL, 1)
    xj = xj.view(batch_size*queryL, 1, sourceL)
    xj_confi = xj_confi.view(batch_size*queryL, 1, sourceL)

    # -> (batch*queryL, sourceL, sourceL)
    term1 = torch.bmm(xi, xj_confi)
    term2 = xj * xj_confi
    funcF = torch.sum(term1-term2, dim=-1)  # -> (batch*queryL, sourceL)
    funcF = funcF.view(batch_size, queryL, sourceL)

    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)

        # Focal attention in text-to-image direction
        # weiContext: (n_image, n_word, d)
        weiContext = func_attention(cap_i_expand, images, opt)
        t2i_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        
        t2i_sim = t2i_sim.mean(dim=1, keepdim=True)

        # Focal attention in image-to-text direction
        # weiContext: (n_image, n_word, d)
        weiContext = func_attention(images, cap_i_expand, opt)
        i2t_sim = cosine_similarity(images, weiContext, dim=2)
        
        i2t_sim = i2t_sim.mean(dim=1, keepdim=True)

        # Overall similarity for image and text
        sim = t2i_sim + i2t_sim
        similarities.append(sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities

#only compute similarity between i-th text and i-th image and return a score vector
def xattn_scoreoffline(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(1, 1, 1)

        # Focal attention in text-to-image direction
        # weiContext: (n_image, n_word, d)
        weiContext = func_attention(cap_i_expand, images[i:i+1], opt)
        t2i_sim = cosine_similarity(cap_i_expand, weiContext, dim=2).unsqueeze(0)
        t2i_sim = t2i_sim.mean(dim=1, keepdim=True)

        # Focal attention in image-to-text direction
        # weiContext: (n_image, n_word, d)
        weiContext = func_attention(images[i:i+1], cap_i_expand, opt)
        i2t_sim = cosine_similarity(images[i:i+1], weiContext, dim=2).unsqueeze(0)
        i2t_sim = i2t_sim.mean(dim=1, keepdim=True)

        # Overall similarity for image and text
        sim = t2i_sim + i2t_sim
        similarities.append(sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities[0]




class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, marginonline=0.2, marginoffline=0, alpha = 0.3, beta = 1.5, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.marginonline = marginonline
        self.marginoffline = marginoffline
        self.alpha = alpha
        self.beta = beta
        self.max_violation = max_violation

    def forward(self, im, s, s_l, imhnoff, shnoff, s_lhnoff, imhnoff2, shnoff2, s_lhnoff2):
        # compute image-sentence score matrix
        scores = xattn_score(im, s, s_l, self.opt)
        pos = scores.diag().view(im.size(0))
        i2thnoff = xattn_scoreoffline(im, shnoff, s_lhnoff, self.opt)
        t2ihnoff = xattn_scoreoffline(imhnoff, s, s_l, self.opt)
        i2thnoff2 = xattn_scoreoffline(imhnoff, shnoff, s_lhnoff, self.opt)
        t2ihnoff2 = xattn_scoreoffline(imhnoff2, shnoff2, s_lhnoff2, self.opt)
        # disable the positive pair
        scores2 = scores - 10*torch.eye(len(scores)).cuda()
        # get the score list of the online hard negative pairs
        i2thnon = scores2.max(1)[0]
        t2ihnon = scores2.max(0)[0]
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.marginonline + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.marginonline + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        cweight = self.beta - (i2thnoff - i2thnon)/self.alpha
        iweight = self.beta - (t2ihnoff - t2ihnon)/self.alpha
        cost_s = cost_s * cweight
        cost_im = cost_im * iweight

        cost_s2 = (self.marginoffline + i2thnoff - pos).clamp(min=0)
        cost_im2 = (self.marginoffline + t2ihnoff - pos).clamp(min=0)
        cost_s3 = (self.marginoffline + i2thnoff2 - pos).clamp(min=0)
        cost_im3 = (self.marginoffline + t2ihnoff2 - pos).clamp(min=0)

        return cost_s.sum() + cost_im.sum()+ cost_s2.sum() + cost_im2.sum() + cost_s3.sum() + cost_im3.sum()


class BFAN(object):
    """
    Bidirectional Focal Attention Network (BFAN) model
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   no_txtnorm=opt.no_txtnorm)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         marginonline=opt.marginonline,
                                         marginoffline=opt.marginoffline,
                                         alpha = opt.alpha,
                                         beta = opt.beta,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens

    def forward_loss(self, img_emb, cap_emb, cap_len, img_embhnoff, cap_embhnoff, cap_lenhnoff, img_embhnoff2, cap_embhnoff2, cap_lenhnoff2,**kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, cap_len, img_embhnoff, cap_embhnoff, cap_lenhnoff, img_embhnoff2, cap_embhnoff2, cap_lenhnoff2)
        self.logger.update('Le', loss.item())
        return loss

    def train_emb(self, images, captions, lengths, ids, imageshnoff, captionshnoff, lengthshnoff, imageshnoff2, captionshnoff2, lengthshnoff2, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens = self.forward_emb(
            images, captions, lengths)
        img_embhnoff, cap_embhnoff, cap_lenshnoff = self.forward_emb(imageshnoff, captionshnoff, lengthshnoff)
        img_embhnoff2, cap_embhnoff2, cap_lenshnoff2 = self.forward_emb(imageshnoff2, captionshnoff2, lengthshnoff2)
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens, img_embhnoff, cap_embhnoff, cap_lenshnoff, img_embhnoff2, cap_embhnoff2, cap_lenshnoff2)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
