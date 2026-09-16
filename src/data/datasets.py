"""
PyTorch Dataset classes for image-text matching (ITM) and masked-language-
modeling (MLM) training on the preprocessed MIMIC-CXR sentence dict produced
by `src.data.preprocessing.build_sentence_dict_from_rows`.
"""
import numpy as np
import torchvision as tv
from PIL import Image
from torch.utils.data import Dataset


MAX_DIM = 256

train_transform = tv.transforms.Compose([
    tv.transforms.Grayscale(num_output_channels=1),
    tv.transforms.Resize((MAX_DIM, MAX_DIM)),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.RandomRotation(10),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transform = tv.transforms.Compose([
    tv.transforms.Grayscale(num_output_channels=1),
    tv.transforms.Resize((MAX_DIM, MAX_DIM)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.5], std=[0.5])
])


class MimicDataset(Dataset):
    def __init__(self, dataset, max_length, transform=train_transform, mode='train'):
        super().__init__()
        self.transform  = transform
        self.mode       = mode
        self.max_length = max_length + 1
        self.datadict   = dataset['data_dict']
        self.vocab_size = dataset['vocab_size']

        if mode == 'train':
            self.keys = dataset['data_split']['train_uids']
        elif mode == 'val':
            self.keys = dataset['data_split']['val_uids']
        elif mode == 'test':
            self.keys = dataset['data_split']['test_uids']
        else:
            raise ValueError(f'mode {mode} not supported')

        self.__sep_id__  = 102
        self.__mask_id__ = 103
        self.__pad_id__  = 0

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        uid   = self.keys[idx]
        entry = self.datadict[uid]

        image = Image.open(entry['image_path']).convert('L')
        if self.transform:
            image = self.transform(image)

        caption = np.array(entry['token_ids'])
        max_len = self.max_length

        max_len_array = np.zeros(max_len, dtype='int')
        cap_mask      = np.zeros(max_len, dtype='int')

        if len(caption) <= max_len:
            cap_mask[:len(caption)]      = 1
            max_len_array[:len(caption)] = caption
        else:
            cap_mask[:]       = 1
            max_len_array     = caption[:max_len]
            max_len_array[-1] = self.__sep_id__

        cap_mask = cap_mask.astype(bool)
        cap_lens = cap_mask.sum(-1)
        image_id = entry['image_id']

        return image, max_len_array, cap_mask, image_id, cap_lens


class MimicDatasetMLM(Dataset):
    def __init__(self, dataset, max_length, transform=train_transform, mode='train'):
        super().__init__()
        self.mode       = mode
        self.max_length = max_length + 1
        self.datadict   = dataset['data_dict']
        self.vocab_size = dataset['vocab_size']

        if mode == 'train':
            self.keys = dataset['data_split']['train_uids']
        elif mode == 'val':
            self.keys = dataset['data_split']['val_uids']
        elif mode == 'test':
            self.keys = dataset['data_split']['test_uids']
        else:
            raise ValueError(f'mode {mode} not supported')

        self.__sep_id__  = 102
        self.__mask_id__ = 103
        self.__pad_id__  = 0

    def __len__(self):
        return len(self.keys)

    def mask_input(self, encoded_texts):
        inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
        inp_mask[encoded_texts <= 3] = False

        labels = -1 * np.ones(encoded_texts.shape, dtype=int)
        labels[inp_mask] = encoded_texts[inp_mask]

        encoded_texts_masked = np.copy(encoded_texts)

        inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
        encoded_texts_masked[inp_mask_2mask] = self.__mask_id__

        inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1/9)
        encoded_texts_masked[inp_mask_2random] = np.random.randint(
            4, self.__mask_id__, inp_mask_2random.sum()
        )

        return encoded_texts_masked, labels

    def __getitem__(self, idx):
        uid   = self.keys[idx]
        entry = self.datadict[uid]

        caption_unmask = np.array(entry['token_ids'])
        caption, label = self.mask_input(caption_unmask)

        max_len       = self.max_length
        max_len_array = np.zeros(max_len, dtype='int')
        cap_mask      = np.zeros(max_len, dtype='int')
        label_full    = -1 * np.ones(max_len, dtype='int')

        if len(caption) <= max_len:
            label_full[:len(label)]      = label
            cap_mask[:len(caption)]      = 1
            max_len_array[:len(caption)] = caption
        else:
            label_full        = label[:max_len]
            label_full[-1]    = -1
            cap_mask[:]       = 1
            max_len_array     = caption[:max_len]
            max_len_array[-1] = self.__sep_id__

        cap_mask = cap_mask.astype(bool)
        cap_lens = cap_mask.sum(-1)
        image_id = entry['image_id']

        return max_len_array, cap_mask, label_full, image_id, cap_lens


def build_dataset_itm(dataset, mode='train', cfg=None, out_dir=None):
    transform = train_transform if mode == 'train' else val_transform
    return MimicDataset(dataset, max_length=cfg.max_length, transform=transform, mode=mode)

def build_dataset_mlm(dataset, mode='train', cfg=None):
    transform = train_transform if mode == 'train' else val_transform
    return MimicDatasetMLM(dataset, max_length=cfg.max_length, transform=transform, mode=mode)


def collate_fn_ignore_none(batch):
    """Drops any None samples (e.g. from a failed image load) before the
    default collate. Used as the DataLoader collate_fn for both ITM and
    MLM loaders."""
    import torch
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
