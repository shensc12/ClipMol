import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import h5py
import random
import re
import os
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


class ChemTokenizer:
    def __init__(self, vocab_file, mode='smiles'):
        self.vocab = {}
        self.ids_to_tokens = {}

        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

        with open(vocab_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        idx = 0
        for line in lines:
            token = line.strip()
            if token.startswith('[source'):
                parts = token.split(']')
                if len(parts) > 1: token = parts[1].strip()

            if token and token not in self.vocab:
                self.vocab[token] = idx
                self.ids_to_tokens[idx] = token
                idx += 1

        self.pad_token = self.vocab.get('<pad>', 0)
        self.unk_token = self.vocab.get('<unk>', 1)
        self.cls_token = self.vocab.get('<cls>', self.vocab.get('<bos>', 2))
        self.sep_token = self.vocab.get('<sep>', 3)
        self.eos_token = self.vocab.get('<eos>', 3)
        if mode == 'inchi':
            self.pattern = re.compile(r'(1[ST]\/|\/[a-z]|[A-Z][a-z]{,1}|[0-9]|[\+\-,;=#:\*@\?\$\.]|[()\[\]])')
        else:
            self.pattern = re.compile(
                r'(\[[^\]]+\]|%[0-9]{2}|[A-Z][a-z]|[A-Z]|[a-z]|[0-9]|[\#\=\-\+\/\\\.\(\)\@])')
    def text_to_sequence(self, text, max_len=None):
        tokens = [t for t in self.pattern.findall(text)]
        ids = [self.cls_token]
        ids.extend([self.vocab.get(t, self.unk_token) for t in tokens])
        ids.append(self.eos_token)

        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = self.eos_token

        return torch.tensor(ids, dtype=torch.long)

    def get_vocab_size(self):
        return len(self.vocab)


class MoleculeDataset(Dataset):
    def __init__(self, h5_path, inchi_tokenizer, smiles_tokenizer, config):
        self.h5_path = h5_path
        self.inchi_tok = inchi_tokenizer
        self.smiles_tok = smiles_tokenizer
        self.config = config

        with h5py.File(h5_path, 'r') as f:
            self.data_key = 'data' if 'data' in f else 'virtual_data'
            if self.data_key not in f:
                raise KeyError(f"Key not found in {h5_path}")
            self.length = f[self.data_key].shape[0]
        self.h5_file = None
        self.dataset = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')
            self.dataset = self.h5_file[self.data_key]

        try:
            row_bytes = self.dataset[idx]
        except Exception:
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')
            self.dataset = self.h5_file[self.data_key]
            row_bytes = self.dataset[idx]

        row_str = [b.decode() for b in row_bytes]
        inchi_str = row_str[1]
        smiles_candidates = row_str[2:]

        if smiles_candidates:
            selected_smiles = random.choice(smiles_candidates)
        else:
            selected_smiles = ""

        inchi_tensor = self.inchi_tok.text_to_sequence(inchi_str, self.config.max_inchi_len)
        smiles_tensor = self.smiles_tok.text_to_sequence(selected_smiles, self.config.max_smiles_len)

        return inchi_tensor, smiles_tensor

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()


class CollateFn:
    def __init__(self, inchi_pad_idx, smiles_pad_idx):
        self.inchi_pad_idx = inchi_pad_idx
        self.smiles_pad_idx = smiles_pad_idx

    def __call__(self, batch):
        # batch: List of (inchi, smiles)
        inchi_list = [item[0] for item in batch]
        smiles_list = [item[1] for item in batch]

        inchi_batch = pad_sequence(inchi_list, batch_first=True, padding_value=self.inchi_pad_idx)
        smiles_batch = pad_sequence(smiles_list, batch_first=True, padding_value=self.smiles_pad_idx)

        return inchi_batch, smiles_batch


def get_dataloader(config, mode='train'):
    config.mode = mode

    inchi_tok = ChemTokenizer(config.inchi_vocab_path, mode='inchi')
    smiles_tok = ChemTokenizer(config.smiles_vocab_path, mode='smiles')

    config.inchi_padding_idx = inchi_tok.pad_token
    config.smiles_padding_idx = smiles_tok.pad_token

    if mode == 'train':
        fpath = config.train_file
    elif mode == 'val':
        fpath = config.val_file
    else:
        fpath = getattr(config, 'test_file', config.val_file)

    dataset = MoleculeDataset(fpath, inchi_tok, smiles_tok, config)

    collate_fn = CollateFn(
        inchi_pad_idx=inchi_tok.pad_token,
        smiles_pad_idx=smiles_tok.pad_token
    )

    sampler = None
    shuffle_bool = (mode == 'train')

    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle_bool)
        dataloader_shuffle = False
    else:
        dataloader_shuffle = shuffle_bool

    num_workers = getattr(config, 'num_workers', 4)
    if dist.is_available() and dist.is_initialized():
        num_workers = max(2, int(num_workers / 4))

    return DataLoader(
        dataset,
        batch_size=config.batch_size,  
        shuffle=dataloader_shuffle,  
        sampler=sampler,  
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2, 
        persistent_workers=True
    ), inchi_tok, smiles_tok
