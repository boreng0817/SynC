import torch
import json
import os
import re
import json
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm
import clip


def simRetrievalGetPath(args):
    source = "".join(args.caption_source)
    similarity_path = f'{args.root}/sim_{args.similarity_metric}_Source{source}_{args.clip_name}_{args.mapping_strategy}{args.K}_{args.retrieval_strategy}{args.i2tK}.npy'
    index_path = f'{args.root}/sim_{args.similarity_metric}_Source{source}_index_{args.clip_name}_{args.mapping_strategy}_{args.retrieval_strategy}{args.i2tK}.npy'
    return similarity_path, index_path

def simClipsGetPath(args):
    similarity_path = f'{args.root}/sim_clips_{args.clip_name}_{args.mapping_strategy}_{args.retrieval_strategy}{args.K}.npy'
    index_path = f'{args.root}/sim_clips_index_{args.clip_name}_{args.mapping_strategy}_{args.retrieval_strategy}.npy'
    return similarity_path, index_path

def loadMetaData(args):

    gt_captions_path  = f'{args.root}/meta_gt_captions.json'
    images_path       = f'{args.root}/meta_images.json'

    with open(gt_captions_path, 'r') as f:
        gt_captions = json.load(f) # Dict{str(image_id) : str(gt caption}}

    with open(images_path, 'r') as f:
        images = json.load(f) # List[str(image_id)]

    return images, gt_captions

def loadTextFeature(path, captions, args):
    print(f'In loadTextFeature: {path}')
    print(args.clip_type)
    if 'sieve' in args.root and 'sbert' in path:
        return loadTextFeatureSBERTSieve(path, captions, args)
    elif 'sbert' in path:
        return loadTextFeatureSBERT(path, captions, args)
    elif 'SigLIP' in path:
        return loadTextFeatureSigLIP(path, captions, args)
    else:
        return loadTextFeatureCLIP(path, captions, args)

@torch.no_grad()
def _utilRetreival(query, key, K=50):
    result = []
    for i in tqdm(range(len(query))):
        q = query[i].unsqueeze(0)
        similarity = q @ key.T
        niber = []
        _, max_id = torch.topk(similarity, k=K)
        niber = max_id[0].tolist()
        result.append(niber)

    return result

@torch.no_grad()
def loadKNNData(captions, args, K=50):
    # return knn per text caption features
    if args.retrieval_strategy == 'i2tClip':
        path = f'{args.root}/knn_text_{args.retrieval_strategy}_source{"".join(args.caption_source)}_{args.clip_name}.json'
    else:
        path = f'{args.root}/knn_text_{args.retrieval_strategy}_{args.clip_name}.json'

    print(f'In loadKNNData: {path}')


    if os.path.exists(path):
        with open(path, 'r') as f:
            knn_text = json.load(f)
    else:
        caption_features_path  = f'{args.root}/gt_caption_features_{args.clip_name}.npy'
        caption_features = loadTextFeature(caption_features_path, captions, args)
        image_features_path = f'{args.root}/image_features_{args.clip_name}.npy'
        image_features = loadImageFeature(image_features_path, args)

        cap_feats = torch.tensor(caption_features).to(args.device)
        im_feats =  torch.tensor(image_features).to(args.device)

        knn_text = []

        if args.retrieval_strategy == 't2tClip':
            knn_text = _utilRetreival(cap_feats, cap_feats)

        elif args.retrieval_strategy == 't2iClip':
            knn_text = _utilRetreival(cap_feats, im_feats)

        elif args.retrieval_strategy == 'i2tClip':
            knn_text = _utilRetreival(im_feats, cap_feats)

        elif args.retrieval_strategy == 'i2iClip':
            knn_text = _utilRetreival(im_feats, im_feats)

        with open(path, 'w') as f:
            json.dump(knn_text, f)

    knn_text = [knn_text[i][:args.K] for i in range(len(knn_text))]
    return knn_text

@torch.no_grad()
def loadTextFeatureSBERT(path, captions, args):
    from sentence_transformers import SentenceTransformer
    if os.path.exists(path):
        text_features = np.load(path)
    else:
        if type(captions) == dict:
            captions = [captions[image] for image in args.images]

        device = args.device
        embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                                          device=device)
        embed_model.eval()
        text_features = []
        batch_size = 256

        for idx in tqdm(range(0, len(captions), batch_size)):
            caption = captions[idx:idx+batch_size]
            embeddings = embed_model.encode(caption)
            text_features.append(embeddings)

        text_features = np.concatenate(text_features)
        text_features /= np.linalg.norm(text_features, axis=1).reshape(-1, 1)

        # save
        np.save(path, text_features)

    return text_features

@torch.no_grad()
def loadTextFeatureCLIP(path, captions, args):

    if os.path.exists(path):
        text_features = np.load(path)
    else:
        if type(captions) == dict:
            captions = [captions[image] for image in args.images]

        device = args.device
        encoder, _ = clip.load(args.clip_type, device)
        text_features = []
        batch_size = 256

        for idx in tqdm(range(0, len(captions), batch_size)):
            caption = captions[idx:idx+batch_size]
            tokens = clip.tokenize(caption, truncate = True).to(device)
            embeddings = encoder.encode_text(tokens).to('cpu').numpy()
            text_features.append(embeddings)

        text_features = np.concatenate(text_features)
        text_features /= np.linalg.norm(text_features, axis=1).reshape(-1, 1)

        # save
        np.save(path, text_features)

    return text_features

@torch.no_grad()
def loadTextFeatureSigLIP(path, captions, args):

    if os.path.exists(path):
        text_features = np.load(path)
    else:
        if type(captions) == dict:
            captions = [captions[image] for image in args.images]

        device = args.device
        model, _ = open_clip.create_model_from_pretrained(args.clip_type, device=device)
        tokenizer = open_clip.get_tokenizer(args.clip_type)
        text_features = []
        batch_size = 256

        for idx in tqdm(range(0, len(captions), batch_size)):
            caption = captions[idx:idx+batch_size]
            tokens = tokenizer(caption).to(device)
            embeddings = model.encode_text(tokens).to('cpu').numpy()
            text_features.append(embeddings)

        text_features = np.concatenate(text_features)
        text_features /= np.linalg.norm(text_features, axis=1).reshape(-1, 1)

        # save
        np.save(path, text_features)

    return text_features

def loadImageFeature(path, args):
    print(f'In loadTextFeature: {path}')
    print(args.clip_type)
    if 'SigLIP' in path:
        return loadImageFeatureSigLIP(path, args)
    else:
        return loadImageFeatureCLIP(path, args)

@torch.no_grad()
def loadImageFeatureSigLIP(path, args):
    print(f'In loadImageFeature: {path}')
    print(args.clip_type)

    if os.path.exists(path):
        image_features = np.load(path)
    else:

        device = args.device
        images = args.images
        encoder, preprocess = open_clip.create_model_from_pretrained(args.clip_type, device=device)
        image_features = []
        batch_size = 64

        for idx in tqdm(range(0, len(images), batch_size)):
            image_paths = [f'{args.root}/images/{image}' for image in images[idx:idx+batch_size]]
            ims = [preprocess(Image.open(image_path)) for image_path in image_paths]
            ims = torch.stack(ims).to(device)
            embeddings = encoder.encode_image(ims).to('cpu').numpy()
            image_features.append(embeddings)

        image_features = np.concatenate(image_features)
        image_features /= np.linalg.norm(image_features, axis=1).reshape(-1, 1)

        # save
        np.save(path, image_features)

    return image_features



@torch.no_grad()
def loadImageFeatureCLIP(path, args):
    print(f'In loadImageFeature: {path}')
    print(args.clip_type)

    if os.path.exists(path):
        image_features = np.load(path)
    else:

        device = args.device
        images = args.images
        encoder, proprecess = clip.load(args.clip_type, device)
        image_features = []
        batch_size = 64

        for idx in tqdm(range(0, len(images), batch_size)):
            image_paths = [f'{args.root}/images/{image}' for image in images[idx:idx+batch_size]]
            ims = [proprecess(Image.open(image_path)) for image_path in image_paths]
            ims = torch.stack(ims).to(device)
            embeddings = encoder.encode_image(ims).to('cpu').numpy()
            image_features.append(embeddings)

        image_features = np.concatenate(image_features)
        image_features /= np.linalg.norm(image_features, axis=1).reshape(-1, 1)

        # save
        np.save(path, image_features)

    return image_features
