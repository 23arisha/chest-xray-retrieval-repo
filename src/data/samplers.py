"""
Custom sampler ensuring each unique training image contributes exactly one
(randomly chosen) sentence per epoch, since a single image can have several
associated findings sentences (see manuscript Sec. 3.1.1).
"""
import random

import torch


class UniqueImageSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.image_to_uids = {}
        for pos, uid in enumerate(dataset.keys):
            img_id = dataset.datadict[uid]['image_id']
            if img_id not in self.image_to_uids:
                self.image_to_uids[img_id] = []
            self.image_to_uids[img_id].append(pos)  # need position, not uid
        self.image_ids = list(self.image_to_uids.keys())
        print(f'UniqueImageSampler: {len(self.image_ids)} unique images')

    def __iter__(self):
        img_ids = self.image_ids.copy()
        random.shuffle(img_ids)
        selected = []
        for img_id in img_ids:
            uid = random.choice(self.image_to_uids[img_id])
            selected.append(uid)
        return iter(selected)

    def __len__(self):
        return len(self.image_ids)

