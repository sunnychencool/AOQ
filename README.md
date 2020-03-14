# Adaptive Offline Quintuplet Loss for Image-Text Matching (AOQ)
PyTorch code of the paper "Adaptive Offline Quintuplet Loss for Image-Text Matching". It is built on top of [VSRN](https://github.com/KunpengLi1994/VSRN) and [BFAN](https://github.com/CrossmodalGroup/BFAN).

[Tianlang Chen](https://www.cs.rochester.edu/u/tchen45/), Jiajun Deng and [Jiebo Luo](https://www.cs.rochester.edu/u/jluo/). "Adaptive Offline Quintuplet Loss for Image-Text Matching", arxiv, 2020. [[pdf](https://arxiv.org/pdf/2003.03669.pdf)]

## Introduction
Existing image-text matching approaches typically leverage triplet loss with online hard negatives to train the model. For each image or text anchor in a training mini-batch, the model is trained to distinguish between a positive and the most confusing negative of the anchor mined from the mini-batch (i.e. online hard negative). This strategy improves the model's capacity to discover fine-grained correspondences and non-correspondences between image and text inputs. However, the above training approach has the following drawbacks: (1) the negative selection strategy still provides limited chances for the model to learn from very hard-to-distinguish cases. (2) The trained model has weak generalization capability from the training set to the testing set. (3) The penalty lacks hierarchy and adaptiveness for hard negatives with different “hardness” degrees. In this paper, we propose solutions by sampling negatives offline from the whole training set. It provides “harder” offline negatives than online hard negatives for the model to distinguish. Based on the offline hard negatives, a quintuplet loss is proposed to improve the model's generalization capability to distinguish positives and negatives. In addition, a novel loss function that combines the knowledge of positives, offline hard negatives and online hard negatives is created. It leverages offline hard negatives as intermediary to adaptively penalize them based on their distance relations to the anchor. We evaluate the proposed training approach on three state-of-the-art image-text models on the MS-COCO and Flickr30K datasets. Significant performance improvements are observed for all the models, demonstrating the effectiveness and generality of the proposed approach. 

![model](/fig/model.png)
![model](/fig/coco.png)
![model](/fig/f30k.png)

## Requirements 
We recommended the following dependencies.

* Python 2.7 
* [PyTorch](http://pytorch.org/) (0.4.1)
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)
* [pycocotools](https://github.com/cocodataset/cocoapi)
* [torchvision]()
* [matplotlib]()


* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```
## VSRN + Adaptive Offline Quintuplet Loss

## Download data

Download the dataset files and pre-trained models. We use splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). 

We follow [bottom-up attention model](https://github.com/peteanderson80/bottom-up-attention) and [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features for fair comparison. More details about data pre-processing (optional) can be found [here](https://github.com/kuanghuei/SCAN/blob/master/README.md#data-pre-processing-optional). All the data needed for reproducing the experiments in the paper, including image features and vocabularies, can be downloaded from [SCAN](https://github.com/kuanghuei/SCAN) by using:

```bash
wget https://scanproject.blob.core.windows.net/scan-data/data.zip
```

You can also get the data from google drive: https://drive.google.com/drive/u/1/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC. We refer to the path of extracted files for `data.zip` as `$DATA_PATH`. 

## Creating Offline Candidates

As desribed in our paper, we perform global similarity score prediction to obtain the offline candidates of each training image/text. This enables the offline hard negative sampling to train the model. We provide two ways to obtain the offline candidates. 

The most convenient way is to direcly download the offline candidate files (MSCOCO + Flickr30K) from [here](https://drive.google.com/drive/folders/1rBUSlCBzRn0yErOkqor9xZF3CQOM5Vsv?usp=sharing) and put them into `./offcandipath` folder (otherwise please modify `--offcandipath`). We created these files by training [VSRN](https://github.com/KunpengLi1994/VSRN) and then perform global similarity score prediction. Otherwise, you can train your own VSRN models, extracting the embeddings of all the training images and captions and utilizing `./tools/geti2toffcandi.py` and `./tools/gett2ioffcandi.py` to create the files.

## Evaluate pre-trained models
Modify the model_paths and data_path in the `eval.py` file. Then run:

```bash
python eval.py
```

To do cross-validation on MSCOCO 1K test set, pass `fold5=True`. Pass `fold5=False` for evaluation on MSCOCO 5K test set. Pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/16n9XH9CDhfKSUI4S0g3_baQj-J1vwxJP?usp=sharing).

## Training new models
Run `trainAOQ.py`:

For MSCOCO:

```bash
python trainAOQ.py --data_path $DATA_PATH --data_name coco_precomp --logger_name runs/coco_VSRN --max_violation --lr_update=10 
```
You can modify the batch size to 124 (originally 128) so that the model can be trained by single GeForce GTX 1080/2080 Ti GPU.

For Flickr30K:

```bash
python trainAOQ.py --data_path $DATA_PATH --data_name f30k_precomp --logger_name runs/flickr_VSRN --max_violation --lr_update 10 --max_len 60
```

## BFAN + Adaptive Offline Quintuplet Loss

You can follow [here](https://github.com/CrossmodalGroup/BFAN) to download the data and then copy the files in `./BFAN` to the same folder. Still, putting the offline candidate files (same files as above) into the folder './offcandipath'. You can directly follow [here](https://github.com/CrossmodalGroup/BFAN) to train the new models but set `--lr_update=10` for all the situations.

Pre-trained models for MSCOCO/Flickr30K can be downloaded from [here](https://drive.google.com/drive/folders/12Bzx6qAd6L-R9GnSbgvvJPpSdcfWG97D?usp=sharing) and evaluated by run `eval.py` in the `./BFAN` folder. All the pre-trained models can achieve the exactly same performance as shown in our paper.


## Reference

If you found this code useful, please cite the following paper:

    @article{chen2020adaptive,
      title={Adaptive Offline Quintuplet Loss for Image-Text Matching},
      author={Chen, Tianlang and Deng, Jiajun and Luo, Jiebo},
      journal={arXiv preprint arXiv:2003.03669},
      year={2020}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)


