import os
import torch
import numpy as np
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import nltk
from .utils import simRetrievalGetPath, simClipsGetPath


def simRetrieval(knn_image, gt_caption_features, args):
    # require Isyn, Cgt embedding
    similarity_path, index_path = simRetrievalGetPath(args)

    if not os.path.exists(similarity_path):
        similarity = []
        index = []
        datasource_caption_features = gt_caption_features

        if args.mapping_strategy == 'oneToOne':
            for i in tqdm(range(len(gt_caption_features))):
                retrieved_captions = knn_image[i]
                retrieved_features = [torch.tensor(datasource_caption_features[n]) for n in retrieved_captions] # gt_caption_features -> datasetore_caption_features
                retrieved_features = torch.stack(retrieved_features, dim=0).to(args.device)

                caption_feature = torch.tensor(gt_caption_features[i]).to(args.device)
                sim = retrieved_features @ caption_feature
                sim = max(sim)
                similarity.append(float(sim.cpu()))
                index.append(i)
                    
        elif args.mapping_strategy in ['oneToMany']:
            retrieved_features = []
            for i in tqdm(range(len(gt_caption_features))):
                neighbor = args.knn_text[i]
                batch_retrieved_features = []
                for neigh in neighbor:
                    retrieved_captions = knn_image[neigh]
                    re_features = [datasource_caption_features[n] for n in retrieved_captions]
                    batch_retrieved_features.append(re_features)
                retrieved_features.append(batch_retrieved_features)

            retrieved_features = torch.tensor(np.array(retrieved_features), dtype=torch.half).to(args.device)
            gt_caption_features = torch.tensor(gt_caption_features, dtype=torch.half).to(args.device)

            for i in tqdm(range(len(gt_caption_features))):
                neighbor = args.knn_text[i]
                sim = retrieved_features[i] @ gt_caption_features[i]# (n, k, dim) (dim)

                sim, _ = torch.max(sim.cpu(), dim=1)

                idx = np.argmax(sim)
                similarity.append(float(sim[idx]))
                index.append(neighbor[idx])

        similarity = np.array(similarity)
        index = np.array(index)
        np.save(similarity_path, similarity)
        np.save(index_path, index)
    else:
        similarity = np.load(similarity_path)
        index = np.load(index_path)

    return list(zip(similarity, index))

def simClips(image_features, gt_caption_features, args):
    # require Isyn, Cgt embedding
    similarity_path, index_path = simClipsGetPath(args)

    if not os.path.exists(similarity_path):
        similarity = []
        index = []
        if args.mapping_strategy == 'oneToOne':
            for i in tqdm(range(len(image_features))):
                sim = image_features[i] @ gt_caption_features[i].T
                similarity.append(sim)
                index.append(i)
                    
        elif args.mapping_strategy in ['oneToMany']:
            im_feats = torch.tensor(image_features).to(args.device)
            cap_feats = torch.tensor(gt_caption_features).to(args.device)
            for i in tqdm(range(len(im_feats))):
                neighbor = args.knn_text[i]
                text_emb = cap_feats[i]
                image_emb = torch.stack([im_feats[neigh] for neigh in neighbor], dim=0)

                sim = text_emb @ image_emb.T
                sim = sim.cpu()
                idx = np.argmax(sim)

                similarity.append(sim[idx])
                index.append(neighbor[idx])

        similarity = np.array(similarity)
        index = np.array(index)
        np.save(similarity_path, similarity)
        np.save(index_path, index)
    else:
        similarity = np.load(similarity_path)
        index = np.load(index_path)

    return list(zip(similarity, index))

