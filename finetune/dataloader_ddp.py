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
            # InChI 专用正则：捕获 1S/, 层标签 /c, 连续数字
            self.pattern = re.compile(r'(1[ST]\/|\/[a-z]|[A-Z][a-z]{,1}|[0-9]|[\+\-,;=#:\*@\?\$\.]|[()\[\]])')
        else:
            # SMILES 专用正则：捕获方括号整体, 单个环数字, 芳香原子
            self.pattern = re.compile(
                r'(\[[^\]]+\]|%[0-9]{2}|[A-Z][a-z]|[A-Z]|[a-z]|[0-9]|[\#\=\-\+\/\\\.\(\)\@])')
    def text_to_sequence(self, text, max_len=None):
        """仅截断，不Padding (由collate_fn处理)"""
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

        # 仅在主进程读取一次长度，不保持句柄
        with h5py.File(h5_path, 'r') as f:
            self.data_key = 'data' if 'data' in f else 'virtual_data'
            if self.data_key not in f:
                raise KeyError(f"Key not found in {h5_path}")
            self.length = f[self.data_key].shape[0]

        # 这里的 h5_file 设为 None，将在 __getitem__ 中延迟初始化
        self.h5_file = None
        self.dataset = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # [优化核心]：Lazy Loading - 每个 Worker 进程只打开一次文件
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')
            self.dataset = self.h5_file[self.data_key]

        # 直接读取，无需反复 open/close
        try:
            row_bytes = self.dataset[idx]
        except Exception:
            # 容错重试 (防止偶尔的 H5 读取冲突)
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
        # 析构时关闭文件
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

        # 动态 Padding
        inchi_batch = pad_sequence(inchi_list, batch_first=True, padding_value=self.inchi_pad_idx)
        smiles_batch = pad_sequence(smiles_list, batch_first=True, padding_value=self.smiles_pad_idx)

        return inchi_batch, smiles_batch


def get_dataloader(config, mode='train'):
    # [关键] 注入 mode，供 Dataset 使用
    config.mode = mode

    # 初始化 Tokenizer (确保路径正确)
    inchi_tok = ChemTokenizer(config.inchi_vocab_path, mode='inchi')
    smiles_tok = ChemTokenizer(config.smiles_vocab_path, mode='smiles')

    # 回写 pad_idx 到 config，供模型使用
    config.inchi_padding_idx = inchi_tok.pad_token
    config.smiles_padding_idx = smiles_tok.pad_token

    # 选择文件路径
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

    # [DDP 核心] 实例化 DistributedSampler
    if dist.is_available() and dist.is_initialized():
        # shuffle=True 表示每个 epoch 会重新随机种子的顺序
        # shuffle=False (用于验证集) 表示顺序固定
        sampler = DistributedSampler(dataset, shuffle=shuffle_bool)

        # 如果用了 Sampler，DataLoader 的 shuffle 必须为 False
        dataloader_shuffle = False
    else:
        # 单卡模式
        dataloader_shuffle = shuffle_bool

    # 计算合适的 workers 数量
    # 如果 config.num_workers 是 16，且有 8 张卡，单机就是 128 个进程，CPU 会扛不住
    # 建议设为 2~4 即可
    num_workers = getattr(config, 'num_workers', 4)
    if dist.is_available() and dist.is_initialized():
        # 简单策略：如果总数很大，这里强制除以 4 或者固定为 4
        # 也可以在 config.py 里直接改小
        num_workers = max(2, int(num_workers / 4))

    return DataLoader(
        dataset,
        batch_size=config.batch_size,  # 注意：这是单卡 Batch Size
        shuffle=dataloader_shuffle,  # DDP 下必须为 False
        sampler=sampler,  # 注入 DistributedSampler
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,  # 降低预取因子以节省内存
        persistent_workers=True
    ), inchi_tok, smiles_tok