import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from rdkit import Chem
from rdkit import RDLogger
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import warnings
import csv
import shutil
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")


from model_lstm_cnn import DualEncoderModel
from dataloader_ddp import ChemTokenizer
from config import Config


# =============================================================================
# 常用多任务数据集的默认列名定义
# =============================================================================
def get_default_tasks(dataset_name):
    d_name = dataset_name.lower()
    if d_name == 'tox21':
        return [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
    elif d_name == 'clintox' or "clintox_42" in d_name:
        return ['FDA_APPROVED', 'CT_TOX_FREE']
    elif d_name == 'sider':
        return [
            'Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues',
            'Eye disorders', 'Investigations', 'Musculoskeletal and connective tissue disorders',
            'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders',
            'Reproductive system and breast disorders',
            'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
            'General disorders and administration site conditions', 'Endocrine disorders',
            'Surgical and medical procedures',
            'Immune system disorders', 'Congenital, familial and genetic disorders', 'Infections and infestations',
            'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders', 'Renal and urinary disorders',
            'Vascular disorders', 'Blood and lymphatic system disorders', 'Nervous system disorders',
            'Skin and subcutaneous tissue disorders', 'Cardiac disorders', 'Ear and labyrinth disorders',
            'Pregnancy, puerperium and perinatal conditions', 'Injury, poisoning and procedural complications'
        ]
    elif d_name == 'muv':
        return []
    elif d_name == 'hiv':
        return ['HIV_active']
    elif d_name == 'bace':
        return ['Class']
    elif d_name == 'bbbp':
        return ['p_np']
    elif 'qm9' in d_name:
        return ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
    else:
        return ['measure']


def generate_inchi_on_the_fly(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return ""
        inchi = Chem.MolToInchi(mol)
        if inchi.startswith("InChI="): inchi = inchi[6:]
        return inchi.replace('"', '').replace("'", "")
    except:
        return ""


class FusionDataset(Dataset):
    def __init__(self, df, smiles_tokenizer, inchi_tokenizer, task_type, measure_name=None, input_type='both',
                 split_name='data', target_mean=None, target_std=None):
        self.smiles_tokenizer = smiles_tokenizer
        self.inchi_tokenizer = inchi_tokenizer
        self.task_type = task_type
        self.input_type = input_type
        self.target_mean = target_mean
        self.target_std = target_std
        self.data = []

        if task_type == 'multitask':
            if isinstance(measure_name, list):
                self.target_cols = measure_name
            else:
                self.target_cols = [measure_name]
        else:
            self.target_cols = [measure_name] if not isinstance(measure_name, list) else measure_name

        df.columns = [str(c).strip().lower() for c in df.columns]
        self.target_cols_lower = [str(c).strip().lower() for c in self.target_cols]

        need_gen_inchi = False
        if (input_type == 'inchi' or input_type == 'both'):
            if 'inchi' not in df.columns:
                print(f"[-] Note: 'inchi' column not found in {split_name}. Generating on-the-fly...")
                need_gen_inchi = True

        print(f"[-] Preprocessing {split_name} data (Target Cols: {len(self.target_cols)} tasks)...")
        valid_count = 0
        total_pos_labels = 0

        iterator = tqdm(df.iterrows(), total=len(df), desc=f"Loading {split_name}", unit="mol")

        for i, row in iterator:
            can_smi = None
            for key in ['smiles', 'smile', 'canonical_smiles']:
                if key in row:
                    can_smi = row[key]
                    break

            if pd.isna(can_smi) or str(can_smi).strip() == "": continue
            can_smi = str(can_smi).strip()

            inchi_str = ""
            if input_type in ['inchi', 'both']:
                if need_gen_inchi:
                    inchi_str = generate_inchi_on_the_fly(can_smi)
                else:
                    val = row.get('inchi', "")
                    inchi_str = str(val) if not pd.isna(val) else ""
                    inchi_str = inchi_str.replace('"', '').replace("'", "")
                    if inchi_str.startswith("InChI="): inchi_str = inchi_str[6:]
                if inchi_str == "": continue

            targets = []
            mask = []

            for col_idx, col_lower in enumerate(self.target_cols_lower):
                val = np.nan
                if col_lower in row:
                    val = row[col_lower]

                if pd.isna(val) or val == '':
                    targets.append(0.0)
                    mask.append(0.0)
                else:
                    try:
                        float_val = float(val)
                        if self.task_type == 'regression' and self.target_mean is not None and self.target_std is not None:
                            float_val = (float_val - self.target_mean[col_idx]) / self.target_std[col_idx]

                        targets.append(float_val)
                        mask.append(1.0)
                        if float_val == 1.0: total_pos_labels += 1
                    except:
                        targets.append(0.0)
                        mask.append(0.0)

            self.data.append({
                'smiles': can_smi,
                'inchi': inchi_str,
                'labels': torch.tensor(targets, dtype=torch.float32),
                'mask': torch.tensor(mask, dtype=torch.float32)
            })
            valid_count += 1

        if valid_count > 0 and total_pos_labels == 0 and task_type != 'regression':
            print(f"[WARNING] No positive labels found in {split_name}! Check column names match: {self.target_cols}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        smiles_list = [item['smiles'] for item in batch]
        inchi_list = [item['inchi'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])

        s_ids = [self.smiles_tokenizer.text_to_sequence(s) for s in smiles_list]
        s_pad = getattr(self.smiles_tokenizer, 'pad_token', 0)
        smiles_tensor = pad_sequence(s_ids, batch_first=True, padding_value=s_pad)

        i_ids = [self.inchi_tokenizer.text_to_sequence(s) for s in inchi_list]
        i_pad = getattr(self.inchi_tokenizer, 'pad_token', 0)
        inchi_tensor = pad_sequence(i_ids, batch_first=True, padding_value=i_pad)

        return smiles_tensor, inchi_tensor, labels, masks

class ModalityGateFusion(nn.Module):
    def __init__(self, smiles_dim, inchi_dim, hidden_dim):
        super().__init__()

        self.proj_s = nn.Linear(smiles_dim, hidden_dim) if smiles_dim != hidden_dim else nn.Identity()
        self.proj_i = nn.Linear(inchi_dim, hidden_dim) if inchi_dim != hidden_dim else nn.Identity()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, s_feat, i_feat):
        s_proj = self.proj_s(s_feat)
        i_proj = self.proj_i(i_feat)
        concat_feat = torch.cat([s_proj, i_proj], dim=-1)
        g = self.gate(concat_feat)
        fused_feat = g * s_proj + (1 - g) * i_proj
        return fused_feat

class PredictionHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, hidden_dim=512, dropout=0.1, use_bn=True):
        super().__init__()

        if num_layers == 0:
            self.layers = None
            self.final = nn.Linear(input_dim, output_dim)
            return

        self.layers = nn.ModuleList()
        current_dim = input_dim

        for i in range(num_layers):
            block = []
            block.append(nn.Linear(current_dim, hidden_dim))

            if use_bn:
                block.append(nn.BatchNorm1d(hidden_dim))

            block.append(nn.GELU())
            block.append(nn.Dropout(dropout))

            self.layers.append(nn.Sequential(*block))
            current_dim = hidden_dim

        self.final = nn.Linear(current_dim, output_dim)
        self.use_initial_skip = (input_dim == hidden_dim)

    def forward(self, x):
        if self.layers is None:
            return self.final(x)

        out = x
        for i, layer in enumerate(self.layers):
            if i == 0 and self.use_initial_skip:
                out = layer(out) + x
            else:
                out = layer(out)

        logits = self.final(out)
        return logits


class FusionFinetuneModel(pl.LightningModule):
    def __init__(self, args, config, s_tokenizer, i_tokenizer, pos_weight=None, target_mean=None, target_std=None):
        super().__init__()
        self.save_hyperparameters(ignore=['config', 's_tokenizer', 'i_tokenizer', 'pos_weight'])
        self.args = args
        self.config = config
        self.s_tokenizer = s_tokenizer
        self.i_tokenizer = i_tokenizer

        self.target_mean = target_mean
        self.target_std = target_std

        print(f"[-] Loading pretrained weights from {args.pretrained_ckpt}")
        full_model = DualEncoderModel(
            inchi_vocab_size=len(i_tokenizer.vocab),
            smiles_vocab_size=len(s_tokenizer.vocab),
            config=config
        )

        if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
            print(f"[-] Loading pretrained weights from {args.pretrained_ckpt}")
            checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            full_model.load_state_dict(new_state_dict, strict=False)
        else:
            print("[-] No valid checkpoint found or explicitly training from scratch. Initializing weights randomly.")

        self.input_type = args.input_type
        if self.input_type in ['smiles', 'both']:
            self.smiles_encoder = full_model.smiles_encoder
        if self.input_type in ['inchi', 'both']:
            self.inchi_encoder = full_model.inchi_encoder

        if args.freeze_backbone:
            if hasattr(self, 'smiles_encoder'):
                for p in self.smiles_encoder.parameters(): p.requires_grad = False
            if hasattr(self, 'inchi_encoder'):
                for p in self.inchi_encoder.parameters(): p.requires_grad = False

        feat_dim = 0
        if self.input_type == 'smiles':
            feat_dim = config.smiles_embed_dim
        elif self.input_type == 'inchi':
            feat_dim = config.inchi_embed_dim
        elif self.input_type == 'both':
            feat_dim = max(config.smiles_embed_dim, config.inchi_embed_dim)
            self.gate_fusion = ModalityGateFusion(
                smiles_dim=config.smiles_embed_dim,
                inchi_dim=config.inchi_embed_dim,
                hidden_dim=feat_dim
            )

        if args.task_type == 'multitask':
            if isinstance(args.measure_name, list):
                self.num_outputs = len(args.measure_name)
            else:
                self.num_outputs = 1
        else:
            self.num_outputs = 1

        print(f"[-] Prediction Head Input Dim: {feat_dim}, Output Dim: {self.num_outputs}")
        self.head = PredictionHead(
            input_dim=feat_dim,
            output_dim=self.num_outputs,
            num_layers=getattr(args, 'num_layers', 2),
            hidden_dim=getattr(args, 'hidden_dim', feat_dim),
            dropout=args.dropout,
            use_bn=getattr(args, 'use_bn', True)
        )

        if args.task_type == 'regression':
            self.loss_fn = nn.MSELoss()
        else:
            if pos_weight is not None:
                self.register_buffer('pos_weight', torch.tensor(pos_weight))
                self.loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_weight)
                print(f"[-] Using Weighted BCE Loss")
            else:
                self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, s_ids, i_ids):
        s_feat, i_feat = None, None

        if self.input_type in ['smiles', 'both']:
            s_feat = self.smiles_encoder(s_ids, padding_idx=self.s_tokenizer.pad_token)
            if len(s_feat.shape) == 3:
                mask = (s_ids != self.s_tokenizer.pad_token).unsqueeze(-1).float()
                s_feat = (s_feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            s_feat = F.normalize(s_feat, p=2, dim=1)

        if self.input_type in ['inchi', 'both']:
            i_feat = self.inchi_encoder(i_ids, padding_idx=self.i_tokenizer.pad_token)
            if len(i_feat.shape) == 3:
                mask = (i_ids != self.i_tokenizer.pad_token).unsqueeze(-1).float()
                i_feat = (i_feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            i_feat = F.normalize(i_feat, p=2, dim=1)
        if self.input_type == 'both':
            combined = self.gate_fusion(s_feat, i_feat)
        elif self.input_type == 'smiles':
            combined = s_feat
        elif self.input_type == 'inchi':
            combined = i_feat

        logits = self.head(combined)
        return logits

    def training_step(self, batch, batch_idx):
        s_ids, i_ids, labels, masks = batch
        logits = self(s_ids, i_ids)
        loss = self._compute_loss(logits, labels, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        s_ids, i_ids, labels, masks = batch
        logits = self(s_ids, i_ids)
        loss = self._compute_loss(logits, labels, masks)
        return {"val_loss": loss, "logits": logits, "labels": labels, "masks": masks}

    def validation_epoch_end(self, outputs):
        logits = torch.cat([x["logits"] for x in outputs]).detach().cpu()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu()
        masks = torch.cat([x["masks"] for x in outputs]).detach().cpu()
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        metrics = {'val_loss': avg_loss}

        if self.args.task_type == 'regression':
            preds = logits.numpy()
            targets = labels.numpy()
            mask_np = masks.numpy()

            if self.target_mean is not None and self.target_std is not None:
                mean_np = np.array(self.target_mean)
                std_np = np.array(self.target_std)
                preds = preds * std_np + mean_np
                targets = targets * std_np + mean_np

            mae_list = []
            rmse_list = []

            task_names = self.args.measure_name if isinstance(self.args.measure_name, list) else [
                self.args.measure_name]

            for i in range(self.num_outputs):
                valid = np.where(mask_np[:, i] == 1)[0]
                if len(valid) < 1: continue

                y_true = targets[valid, i]
                y_pred = preds[valid, i]

                task_mae = mean_absolute_error(y_true, y_pred)
                task_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                mae_list.append(task_mae)
                rmse_list.append(task_rmse)

                t_name = task_names[i] if i < len(task_names) else f"task_{i}"
                metrics[f'val_mae_{t_name}'] = task_mae

            if len(mae_list) > 0:
                metrics['val_mae'] = np.mean(mae_list)
                metrics['val_rmse'] = np.mean(rmse_list)

                valid_mask = mask_np == 1
                if valid_mask.sum() > 0:
                    metrics['val_r2'] = r2_score(targets[valid_mask], preds[valid_mask])
                else:
                    metrics['val_r2'] = 0.0
            else:
                metrics['val_mae'] = float('inf')
                metrics['val_rmse'] = float('inf')
                metrics['val_r2'] = 0.0
        else:
            probs = torch.sigmoid(logits).numpy()
            targets = labels.numpy()
            mask_np = masks.numpy()
            roc_list, acc_list = [], []
            for i in range(self.num_outputs):
                valid = np.where(mask_np[:, i] == 1)[0]
                if len(valid) < 1: continue

                y_true = targets[valid, i]
                y_score = probs[valid, i]

                if len(np.unique(y_true)) > 1:
                    roc_list.append(roc_auc_score(y_true, y_score))

                acc_list.append(accuracy_score(y_true, (y_score > 0.5).astype(int)))

            if len(roc_list) > 0:
                metrics['val_roc_auc'] = np.mean(roc_list)
            else:
                metrics['val_roc_auc'] = 0.5

            if len(acc_list) > 0:
                metrics['val_accuracy'] = np.mean(acc_list)
            else:
                metrics['val_accuracy'] = 0.0

        print(f"\n[Epoch {self.current_epoch}] Validation Metrics: {metrics}")
        self.log_dict(metrics, prog_bar=True)

    def _compute_loss(self, logits, labels, masks):
        if self.args.task_type == 'regression':
            loss_mat = self.loss_fn(logits, labels)
            total_mask = masks.sum() + 1e-9
            return (loss_mat * masks).sum() / total_mask
        else:
            loss_mat = self.loss_fn(logits, labels)
            total_mask = masks.sum() + 1e-9
            return (loss_mat * masks).sum() / total_mask

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: float(epoch + 1) / 5 if epoch < 5 else 0.95 ** (epoch - 5)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_ckpt', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--inchi_vocab', type=str, default='inchi_vocab.txt')
    parser.add_argument('--smiles_vocab', type=str, default='smiles_vocab.txt')
    parser.add_argument('--input_type', type=str, default='both', choices=['smiles', 'inchi', 'both'])

    parser.add_argument('--task_type', type=str, default='regression')
    parser.add_argument('--dataset_name', type=str, default='custom')
    parser.add_argument('--measure_name', type=str, default='measure')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--metric', type=str, default='loss')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of hidden layers in the prediction head. 0 means linear probe.')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Dimension of each hidden layer in the prediction head.')
    parser.add_argument('--use_bn', action='store_true',
                        help='Enable Batch Normalization in the prediction head.')

    args = parser.parse_args()
    pl.seed_everything(42)

    if args.task_type == 'multitask' and args.measure_name == 'measure':
        print(f"[-] Detecting default tasks for {args.dataset_name}...")
        tasks = get_default_tasks(args.dataset_name)
        if len(tasks) > 0:
            args.measure_name = tasks
            print(f"    -> Loaded {len(tasks)} tasks: {tasks}")
        else:
            print(
                f"[!] Warning: No default tasks found for {args.dataset_name}. Please specify --measure_name manually.")
            args.measure_name = [args.measure_name]
    elif args.task_type == 'multitask' and isinstance(args.measure_name, str):
        args.measure_name = args.measure_name.split(',')
    elif args.task_type == 'regression' and 'qm9' in args.dataset_name.lower() and args.measure_name == 'measure':
        print(f"[-] Detecting default regression tasks for {args.dataset_name}...")
        args.measure_name = get_default_tasks(args.dataset_name)
        args.task_type = 'regression'
        print(f"    -> Loaded {len(args.measure_name)} tasks: {args.measure_name}")

    config = Config()
    if os.path.exists(args.smiles_vocab): config.smiles_vocab_path = args.smiles_vocab
    if os.path.exists(args.inchi_vocab): config.inchi_vocab_path = args.inchi_vocab

    s_tokenizer = ChemTokenizer(config.smiles_vocab_path, mode='smiles')
    i_tokenizer = ChemTokenizer(config.inchi_vocab_path, mode='inchi')

    def get_df(split):
        path = os.path.join(args.data_root, f"{args.dataset_name}_{split}.csv")

        if not os.path.exists(path):
            path = os.path.join(args.data_root, f"{split}.csv")

        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[Error] Failed to read CSV {path}: {e}")
            raise e

    print(f"[-] Loading Data... Input Mode: {args.input_type}")
    train_df = get_df('train')
    valid_df = get_df('valid')

    if len(train_df) == 0:
        raise ValueError("Train DataFrame is empty! Please check CSV file integrity.")

    target_mean, target_std = None, None
    if args.task_type == 'regression':
        print("[-] Calculating Mean and Std for regression targets from training data...")
        target_cols_lower = [str(c).strip().lower() for c in
                             (args.measure_name if isinstance(args.measure_name, list) else [args.measure_name])]

        train_targets = train_df[target_cols_lower].apply(pd.to_numeric, errors='coerce')
        target_mean = train_targets.mean().values
        target_std = train_targets.std().values

        target_std = np.nan_to_num(target_std, nan=1.0)
        target_std[target_std == 0] = 1.0
        target_mean = np.nan_to_num(target_mean, nan=0.0)

        print(f"    -> Mean: {np.round(target_mean, 4)}")
        print(f"    -> Std : {np.round(target_std, 4)}")

    train_ds = FusionDataset(train_df, s_tokenizer, i_tokenizer, args.task_type, args.measure_name, args.input_type,
                             'TRAIN', target_mean=target_mean, target_std=target_std)
    valid_ds = FusionDataset(valid_df, s_tokenizer, i_tokenizer, args.task_type, args.measure_name, args.input_type,
                             'VALID', target_mean=target_mean, target_std=target_std)

    pos_weight = None
    if args.task_type != 'regression':
        print("[-] Calculating class weights for imbalanced data...")
        labels_list = []
        for i in range(len(train_ds)):
            labels_list.append(train_ds[i]['labels'].numpy())
        labels_np = np.stack(labels_list)

        num_pos = np.sum(labels_np == 1, axis=0)
        num_neg = np.sum(labels_np == 0, axis=0)

        num_pos = np.clip(num_pos, a_min=1, a_max=None)
        weights = num_neg / num_pos
        pos_weight = weights.tolist()
        pos_weight = [min(w, 20.0) for w in pos_weight]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_ds.collate_fn, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=valid_ds.collate_fn, num_workers=4)

    model = FusionFinetuneModel(args, config, s_tokenizer, i_tokenizer, pos_weight=pos_weight,
                                target_mean=target_mean, target_std=target_std)

    if args.metric == 'roc_auc':
        monitor, mode = 'val_roc_auc', 'max'
    elif args.metric == 'rmse':
        monitor, mode = 'val_rmse', 'min'
    elif args.metric == 'mae':
        monitor, mode = 'val_mae', 'min'
    else:
        monitor, mode = 'val_loss', 'min'

    logger = pl.loggers.CSVLogger("logs_fusion", name=args.dataset_name)
    progress_bar = TQDMProgressBar(refresh_rate=10)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        dirpath=f"checkpoints/{args.dataset_name}",
        filename="best-checkpoint-{epoch:02d}-{" + monitor + ":.4f}",
        save_top_k=1,
        mode=mode,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpu,
        callbacks=[progress_bar, checkpoint_callback, lr_monitor],
        logger=logger,
        default_root_dir=f'logs_{args.input_type}',
        enable_checkpointing=True
    )

    trainer.fit(model, train_loader, valid_loader)

    print(f"\n[-] Training Finished.")

    best_score = 0.0
    best_model_path = "No_Model_Saved"

    if checkpoint_callback.best_model_score is not None:
        best_score = float(checkpoint_callback.best_model_score.cpu().item())
        best_model_path = checkpoint_callback.best_model_path
        print(
            f"[-] Best Metric ({monitor}): {best_score:.4f} (Epoch {checkpoint_callback.best_model_path.split('epoch=')[1][:2]})")
        print(f"[-] Best Model saved at: {best_model_path}")
    else:
        print(f"[!] Warning: Metric {monitor} not found or model not saved.")

    os.makedirs("result", exist_ok=True)

    summary_file = f"result/{args.dataset_name}_gate_all_results.csv"
    file_exists = os.path.exists(summary_file)

    with open(summary_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'Dataset', 'Input_Type', 'Task', 'Target',
                'Best_Model_Path', 'Metric_Name', 'Best_Score',
                'LR', 'Dropout', 'BatchSize', 'HiddenDim', 'NumLayers',
                'Log_Dir'
            ])

        writer.writerow([
            args.dataset_name, args.input_type, args.task_type,
            str(args.measure_name)[:50],
            best_model_path,
            args.metric, f"{best_score:.4f}",
            args.learning_rate, args.dropout, args.batch_size, args.hidden_dim, args.num_layers,
            logger.log_dir if logger else "No_Log"
        ])
    print(f"[-] Summary and Hyperparameters saved to {summary_file}")


if __name__ == '__main__':
    main()