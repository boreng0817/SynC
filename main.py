import os
import clip
import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
from src.similarity_calculator import simClips, simRetrieval
from src.utils import loadMetaData, loadTextFeature, loadImageFeature, loadKNNData
from src.dataset_builder import datasetBuilder

def main(args):
    print(args.clip_type, args.dataset, args.similarity_metric, args.mapping_strategy, args.retrieval_strategy, args.model, args.K)
    print('='*80)

    if args.similarity_metric in ['retrieval'] :
        gt_caption_features_path  = f'{args.root}/gt_caption_features_sbert.npy'
    else:
        gt_caption_features_path  = f'{args.root}/gt_caption_features_{args.clip_name}.npy'

    image_features_path = f'{args.root}/image_features_{args.clip_name}.npy'

    images, gt_captions = loadMetaData(args)
    args.images = images
    args.gt_captions = gt_captions

    # load embeddings (normalized)
    gt_caption_features  = loadTextFeature(gt_caption_features_path, gt_captions, args)
    image_features       = loadImageFeature(image_features_path, args)

    if args.mapping_strategy == 'oneToMany':
        if args.retrieval_strategy in ['i2iClip', 'i2tClip']:
            knn_image = loadKNNData(gt_captions, args)
            args.knn_image = knn_image
            args.knn_text = knn_image
        else:
            knn_text = loadKNNData(gt_captions, args)
            args.knn_text = knn_text

    if args.similarity_metric == 'clips':
        similarity = simClips(image_features, gt_caption_features, args)
    elif args.similarity_metric == 'retrieval':
        temp_K = args.K
        temp_retrieval_strategy = args.retrieval_strategy
        args.K = args.i2tK
        args.retrieval_strategy = 'i2tClip'
        knn_image = loadKNNData(images, args)
        args.K = temp_K
        args.retrieval_strategy = temp_retrieval_strategy

        similarity = simRetrieval(knn_image, gt_caption_features, args)

    datasetBuilder(images, image_features, gt_captions, similarity, args)

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', help='designate gpu number')
    parser.add_argument('--clip_type', type=str, default='hf-hub:timm/ViT-B-16-SigLIP2-256', help='clip variant', 
                        choices = ['ViT-B/32', 'RN50x4', 'ViT-L/14', 
                                   'hf-hub:timm/ViT-B-16-SigLIP', 'hf-hub:timm/ViT-L-16-SigLIP-384',
                                   'hf-hub:timm/ViT-B-16-SigLIP2-256', 'hf-hub:timm/ViT-L-16-SigLIP2-512'
                                   ])
    parser.add_argument('--dataset_clip_type', type=str, default='ViT-B/32', help='clip variant', 
                        choices = ['ViT-B/32', 'RN50x4', 'RN50x64', 'ViT-L/14'])
    parser.add_argument('--model', type=str, default='PCM-Net', help='dataset for model',
                        choices = ['IFCap', 'ViECap', 'CapDec', 'Smallcap', 'PCM-Net'])
    parser.add_argument('--caption_source', type=list, default=['coco'], help='datastore domain', 
            choices=[['coco'], ['flickr30k'], ['cc3m'], ['coco', 'cc3m']])

    parser.add_argument('--dataset', type=str, default='pcmnet', help='dataset name',
                        choices = ['pcmnet'])
    parser.add_argument('--similarity_metric', type=str, default='retrieval', help='similarity function',
                        choices = ['clips', 'retrieval'])
    parser.add_argument('--mapping_strategy', type=str, default='oneToMany', help='mapping strategy',
                        choices = ['oneToOne', 'oneToMany'])
    parser.add_argument('--retrieval_strategy', type=str, default='t2iClip', help='retrieval strategy',
                        choices = ['t2tClip', 't2iClip', 'i2iClip', 'i2tClip'])
    parser.add_argument('--K', type=int, default=15, help='the number of neighborhoods')
    parser.add_argument('--i2tK', type=int, default=2, help='the number of neighborhoods for i2t')

    args = parser.parse_args()
    args.clip_name = args.clip_type.replace('/', '')
    args.root = f'annotations/{args.dataset}'


    if args.mapping_strategy == 'oneToOne':
        args.K = ''

    main(args)

