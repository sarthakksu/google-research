import numpy as np
import random
import re
import pickle
import os
import copy
import sys
class NERLoader:

    def load(self, texts, _labels, labels2idx, tokenizer ,max_position_embeddings=512):

        idx2labels = {v: k for k, v in labels2idx.items()}
        original_texts = copy.deepcopy(texts)
        label_PAD = labels2idx["[PAD]"]
        text_ids = []
        labels = []
        masks=[]
        for _,(text,label) in enumerate(zip(texts,_labels)):
            textlist = text.split()
            labellist = label.split()
            text_ids_=[]
            labels_ = []
            masks_=[]
            for _,(word,label) in enumerate(zip(textlist,labellist)):
                token = tokenizer.tokenize(word)
                text_ids_.extend(token)

                for i,_ in enumerate(token):
                    if i==0:
                        labels_.append(labels2idx[label])
                        masks_.append(1)
                    else:
                        labels_.append(labels2idx["O"] if label == "O" else labels2idx["I-"+label.split("-")[1]])
                        masks_.append(1)
            labels_=labels_[:max_position_embeddings-2]
            text_ids_=text_ids_[:max_position_embeddings-2]
            masks_ = masks_[:max_position_embeddings-2]
            labels_ = [labels2idx["[CLS]"]]+labels_+[labels2idx["[SEP]"]]
            masks_ = [1]+masks_+[1]
            text_ids_ = tokenizer.convert_tokens_to_ids(["[CLS]"]+text_ids_+["[SEP]"])
            labels.append(labels_)
            text_ids.append(text_ids_)
            masks.append(masks_)#([1]*len(text_ids_))
        PAD = 0
        i=0

        while i < len(text_ids):
            
            while len(text_ids[i])< max_position_embeddings:
                text_ids[i].append(PAD)
                labels[i].append(label_PAD)
                masks[i].append(0)
            
            i += 1
        return text_ids,labels,masks
