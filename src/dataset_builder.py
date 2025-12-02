import os
import json
import pickle
import nltk
import numpy as np
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from src.utils import loadTextFeature, loadImageFeature

def image_counts(n, ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][8:9], make_all=True):
    # return [50_000] + [i for i in range(100_000, n, 100_000)] + [n]
    if make_all:
        return [int(n*r) for r in ratio]
    else:
        return [n]

def loadEntities(captions, args, path='gt_caption_entities'):
    path = f'{args.root}/{path}.json'
    if os.path.exists(path):
        entities = json.load(open(path, 'r'))
    else:
        entities = []
        lemmatizer = WordNetLemmatizer()

        for caption in tqdm(captions):
            detected_entities = []
            pos_tags = nltk.pos_tag(nltk.word_tokenize(caption)) # [('woman': 'NN'), ...]
            for entities_with_pos in pos_tags:
                if entities_with_pos[1] in ['NN', 'NNS']:
                    entity = lemmatizer.lemmatize(entities_with_pos[0].lower().strip())
                    detected_entities.append(entity)
            detected_entities = list(set(detected_entities))
            entities.append(detected_entities)
        json.dump(entities, open(path, 'w'))

    return entities


def buildIFCap(image_features, captions, similarity, image_ids, args):
    # format
    # List[List[str(entity)], str(sentence), torch.tensor]
    
    entities = loadEntities(captions, args)

    dataset = list(zip(entities, captions, image_features, image_ids, similarity))
    dataset.sort(key=lambda x: x[4], reverse=True)

    counts = image_counts(len(dataset))
    thresholds = [f'{dataset[i-1][-1]:.3f}' for i in counts]

    dataset = [list(l[:-1]) for l in dataset]

    for th, length in zip(thresholds, counts):
        file_prefix = f'{args.output_path}_th{th}_length{length}'
        print(file_prefix + '.pickle')
        with open(file_prefix + '.pickle', 'wb') as outfile:
            pickle.dump(dataset[:length], outfile)

    return

def buildViECap(image_features, captions, similarity, args):
    # format
    # List[List[str(entity)], str(sentence), torch.tensor]
    
    entities = loadEntities(captions, args)

    dataset = list(zip(entities, captions, image_features, similarity))
    dataset.sort(key=lambda x: x[3], reverse=True)

    counts = image_counts(len(dataset))
    thresholds = [f'{dataset[i-1][-1]:.3f}' for i in counts]

    dataset = [list(l[:-1]) for l in dataset]

    for th, length in zip(thresholds, counts):
        file_prefix = f'{args.output_path}_th{th}_length{length}'
        print(file_prefix + '.pickle')
        with open(file_prefix + '.pickle', 'wb') as outfile:
            pickle.dump(dataset[:length], outfile)

    return

def buildCapDec():
    return

def buildSmallcap(captions, images_id, similarity, args):
    # format
    # Dict[str(image_id) : str(caption)]
    dataset = list(zip(images_id, captions, similarity))
    dataset.sort(key=lambda x: x[2], reverse=True)

    counts = image_counts(len(dataset))
    thresholds = [f'{dataset[i-1][-1]:.3f}' for i in counts]

    dataset = [list(l[:-1]) for l in dataset]

    for th, length in zip(thresholds, counts):
        file_prefix = f'{args.output_path}_th{th}_length{length}'
        ret = {}
        for i in range(length):
            image_id, caption = dataset[i]
            if image_id not in ret:
                ret[image_id] = []
            ret[image_id].append(caption)
        print(file_prefix + '.pickle')
        with open(file_prefix + '.json', 'w') as outfile:
            json.dump(ret, outfile)

    return


def buildPcmnet(gt_captions_mapped, images_id, similarity, gt_captions, args, use_clipscore=True):
    # format
    # List[str(caption), str(image_id)]
    clip_name = args.dataset_clip_type.replace('/', '')
    pcmnet_format = pickle.load(open(f'{args.root}/mscoco_caption_anno_train_{clip_name}.pkl', 'rb'))

    if use_clipscore:
        image_features_path = f'{args.root}/image_features_{args.clip_name}.npy'
        caption_features_path = f'{args.root}/gt_caption_features_{args.clip_name}.npy'
        image_features = loadImageFeature(image_features_path, args)
        text_features = loadTextFeature(caption_features_path, [], args)
        image_features_di = dict(zip(args.images, image_features))
        clips = [float(5*(text_features[i] @ image_features_di[image])) for i, image in enumerate(images_id)]

    for i, record in enumerate(pcmnet_format):
        image_id = images_id[i] # 'root/image_id.jpg'
        record['image_id'] = image_id.split('/')[-1].split('.')[0]
        if use_clipscore:
            record['clipscore'] = clips[i]
        else:
            record['clipscore'] = 1.0



    dataset = list(zip(pcmnet_format, similarity))
    dataset.sort(key=lambda x: x[1], reverse=True)

    counts = image_counts(len(dataset))
    thresholds = [f'{dataset[i-1][-1]:.3f}' for i in counts]

    dataset = [l[0] for l in dataset]

    for th, length in zip(thresholds, counts):
        prefix_clips = "" if use_clipscore else "_CLIPS1"
        file_prefix = f'{args.output_path}{prefix_clips}_th{th}_length{length}'
        print(file_prefix + '.pickle')
        with open(file_prefix + '.pickle', 'wb') as outfile:
            pickle.dump(dataset[:length], outfile)

    return

def datasetBuilder(images, image_features, gt_captions, similarity, args):

    similarity, index = zip(*similarity)

    if args.dataset_clip_type != args.clip_type and args.model in ['IFCap', 'ViECap', 'CapDec']:
        temp_clip_type = args.clip_type
        args.clip_type = args.dataset_clip_type

        clip_name = args.dataset_clip_type.replace('/', '')
        image_features_path = f'{args.root}/image_features_{clip_name}.npy'
        image_features = loadImageFeature(image_features_path, args)

        args.clip_type = temp_clip_type

    if args.retrieval_strategy in ['t2tClip', 't2iClip']:
        image_features = np.array([image_features[index[i]] for i in range(len(image_features))])
        args.images_id = [args.images[index[i]] for i in range(len(image_features))]
        print('image count', len(set(args.images_id)))
        gt_captions_mapped = [gt_captions[image] for image in images]

    elif args.retrieval_strategy in ['i2tClip', 'i2iClip']:
        args.images_id = [args.images[i] for i in range(len(image_features))]
        print('caption count', len(set(index)))
        gt_captions_mapped = [gt_captions[image] for image in images]
        gt_captions_mapped = [gt_captions_mapped[i] for i in index]

    output_path = f'{args.model}_{args.dataset}_{args.similarity_metric}_{args.K}_{args.i2tK}_{args.mapping_strategy}_{args.retrieval_strategy}_{args.clip_name}'
    args.output_path = output_path

    if args.model == 'IFCap':
        buildIFCap(image_features, gt_captions_mapped, similarity, args.images_id, args)
    elif args.model == 'ViECap':
        buildViECap(image_features, gt_captions_mapped, similarity, args)
    elif args.model == 'CapDec':
        buildViECap(image_features, gt_captions_mapped, similarity, args)
    elif args.model == 'Smallcap':
        buildSmallcap(gt_captions_mapped, args.images_id, similarity, args)
    elif args.model == 'PCM-Net':
        buildPcmnet(gt_captions_mapped, args.images_id, similarity, gt_captions, args)

    return
