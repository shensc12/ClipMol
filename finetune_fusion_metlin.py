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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import csv
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

from model_lstm_cnn import DualEncoderModel
from dataloader_ddp import ChemTokenizer
from config import Config


def get_default_tasks(dataset_name):
    d_name = dataset_name.lower()
    if d_name == 'metlin':
        return ['CCS']
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

        if isinstance(measure_name, list):
            self.target_cols = measure_name
        else:
            self.target_cols = [measure_name]

        df.columns = [str(c).strip().lower() for c in df.columns]
        self.target_cols_lower = [str(c).strip().lower() for c in self.target_cols]

        self.adduct_vocab = ['[m+h]+', '[m-h]-', '[m+na]+']

        need_gen_inchi = False
        if (input_type == 'inchi' or input_type == 'both'):
            if 'inchi' not in df.columns:
                need_gen_inchi = True

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

            adduct_val = str(row.get('adduct', '')).strip().lower()
            if adduct_val in self.adduct_vocab:
                adduct_idx = self.adduct_vocab.index(adduct_val)
            else:
                adduct_idx = 3

            adduct_tensor = torch.zeros(4, dtype=torch.float32)
            adduct_tensor[adduct_idx] = 1.0

            targets, mask = [], []
            for col_idx, col_lower in enumerate(self.target_cols_lower):
                val = row.get(col_lower, np.nan)
                if pd.isna(val) or val == '':
                    targets.append(0.0)
                    mask.append(0.0)
                else:
                    try:
                        float_val = float(val)
                        if self.task_type == 'regression' and self.target_mean is not None:
                            float_val = (float_val - self.target_mean[col_idx]) / self.target_std[col_idx]
                        targets.append(float_val)
                        mask.append(1.0)
                    except:
                        targets.append(0.0)
                        mask.append(0.0)

            self.data.append({
                'smiles': can_smi,
                'inchi': inchi_str,
                'adduct': adduct_tensor,
                'labels': torch.tensor(targets, dtype=torch.float32),
                'mask': torch.tensor(mask, dtype=torch.float32)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        smiles_list = [item['smiles'] for item in batch]
        inchi_list = [item['inchi'] for item in batch]
        adducts = torch.stack([item['adduct'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])

        s_ids = [self.smiles_tokenizer.text_to_sequence(s) for s in smiles_list]
        smiles_tensor = pad_sequence(s_ids, batch_first=True,
                                     padding_value=getattr(self.smiles_tokenizer, 'pad_token', 0))
        i_ids = [self.inchi_tokenizer.text_to_sequence(s) for s in inchi_list]
        inchi_tensor = pad_sequence(i_ids, batch_first=True,
                                    padding_value=getattr(self.inchi_tokenizer, 'pad_token', 0))

        return smiles_tensor, inchi_tensor, adducts, labels, masks



class ModalityGateFusion(nn.Module):
    def __init__(self, smiles_dim, inchi_dim, hidden_dim):
        super().__init__()
        self.proj_s = nn.Linear(smiles_dim, hidden_dim) if smiles_dim != hidden_dim else nn.Identity()
        self.proj_i = nn.Linear(inchi_dim, hidden_dim) if inchi_dim != hidden_dim else nn.Identity()
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

    def forward(self, s_feat, i_feat):
        s_p, i_p = self.proj_s(s_feat), self.proj_i(i_feat)
        g = self.gate(torch.cat([s_p, i_p], dim=-1))
        return g * s_p + (1 - g) * i_p


# =============================================================================
# 5. 预测头 & Lightning Module
# =============================================================================
class PredictionHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, hidden_dim=512, dropout=0.1, use_bn=True):
        super().__init__()
        self.layers = nn.ModuleList()
        curr = input_dim
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(nn.Linear(curr, hidden_dim), nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
                              nn.GELU(), nn.Dropout(dropout)))
            curr = hidden_dim
        self.final = nn.Linear(curr, output_dim)

    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return self.final(x)


class FusionFinetuneModel(pl.LightningModule):
    def __init__(self, args, config, s_tokenizer, i_tokenizer, target_mean=None, target_std=None):
        super().__init__()
        self.save_hyperparameters(ignore=['config', 's_tokenizer', 'i_tokenizer'])
        self.args, self.config, self.s_tokenizer, self.i_tokenizer = args, config, s_tokenizer, i_tokenizer
        self.target_mean, self.target_std = target_mean, target_std

        full_model = DualEncoderModel(inchi_vocab_size=len(i_tokenizer.vocab), smiles_vocab_size=len(s_tokenizer.vocab),
                                      config=config)
        checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        full_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)

        self.smiles_encoder = full_model.smiles_encoder if args.input_type in ['smiles', 'both'] else None
        self.inchi_encoder = full_model.inchi_encoder if args.input_type in ['inchi', 'both'] else None

        base_dim = 0
        if args.input_type == 'smiles':
            base_dim = config.smiles_embed_dim
        elif args.input_type == 'inchi':
            base_dim = config.inchi_embed_dim
        elif args.input_type == 'both':
            base_dim = max(config.smiles_embed_dim, config.inchi_embed_dim)
            self.gate_fusion = ModalityGateFusion(config.smiles_embed_dim, config.inchi_embed_dim, base_dim)

        self.num_outputs = len(args.measure_name) if isinstance(args.measure_name, list) else 1
        self.head = PredictionHead(input_dim=base_dim + 4, output_dim=self.num_outputs, num_layers=args.num_layers,
                                   hidden_dim=args.hidden_dim, dropout=args.dropout, use_bn=args.use_bn)
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, s_ids, i_ids, adducts):
        feats = []
        if self.smiles_encoder:
            s = self.smiles_encoder(s_ids, padding_idx=self.s_tokenizer.pad_token)
            if len(s.shape) == 3:
                m = (s_ids != self.s_tokenizer.pad_token).unsqueeze(-1).float()
                s = (s * m).sum(1) / m.sum(1).clamp(1e-9)
            feats.append(F.normalize(s, p=2, dim=1))

        if self.inchi_encoder:
            i = self.inchi_encoder(i_ids, padding_idx=self.i_tokenizer.pad_token)
            if len(i.shape) == 3:
                m = (i_ids != self.i_tokenizer.pad_token).unsqueeze(-1).float()
                i = (i * m).sum(1) / m.sum(1).clamp(1e-9)
            feats.append(F.normalize(i, p=2, dim=1))

        if self.args.input_type == 'both':
            mol_feat = self.gate_fusion(feats[0], feats[1])
        else:
            mol_feat = feats[0]
        return self.head(torch.cat([mol_feat, adducts], dim=1))

    def training_step(self, batch, batch_idx):
        s, i, a, l, m = batch
        loss = (self.loss_fn(self(s, i, a), l) * m).sum() / (m.sum() + 1e-9)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        s, i, a, l, m = batch
        logits = self(s, i, a)
        loss = (self.loss_fn(logits, l) * m).sum() / (m.sum() + 1e-9)
        return {"val_loss": loss, "logits": logits, "labels": l, "masks": m, "adducts": a}

    def validation_epoch_end(self, outputs):
        logits = torch.cat([x["logits"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        masks = torch.cat([x["masks"] for x in outputs]).detach().cpu().numpy()
        adducts = torch.cat([x["adducts"] for x in outputs]).detach().cpu().numpy()

        if self.target_mean is not None:
            logits = logits * self.target_std + self.target_mean
            labels = labels * self.target_std + self.target_mean

        def calc_metrics(y_true, y_pred):
            if len(y_true) < 2: return 0.0, 0.0, 0.0
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            medare = np.median(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
            return r2, rmse, medare

        valid_idx = np.where(masks[:, 0] == 1)[0]
        g_r2, g_rmse, g_medare = calc_metrics(labels[valid_idx, 0], logits[valid_idx, 0])
        metrics = {'val_loss': torch.stack([x["val_loss"] for x in outputs]).mean(), 'val_r2': g_r2, 'val_rmse': g_rmse,
                   'val_medare': g_medare}


        adduct_names = ['MH', 'M_H', 'MNa']  # 代表 M+H, M-H, M+Na
        for i, name in enumerate(adduct_names):
            idx = np.where((masks[:, 0] == 1) & (adducts[:, i] == 1))[0]
            r2, rmse, medare = calc_metrics(labels[idx, 0], logits[idx, 0])
            metrics[f'val_{name}_r2'] = r2
            metrics[f'val_{name}_rmse'] = rmse
            metrics[f'val_{name}_medare'] = medare

        print(f"\n[Epoch {self.current_epoch}] Metrics: {metrics}")
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        sch = LambdaLR(opt, lr_lambda=lambda e: (e + 1) / 5 if e < 5 else 0.95 ** (e - 5))
        return [opt], [sch]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_ckpt', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)

    # 补回这四个被漏掉的参数
    parser.add_argument('--smiles_vocab', type=str, default='smiles_vocab.txt')
    parser.add_argument('--inchi_vocab', type=str, default='inchi_vocab.txt')
    parser.add_argument('--task_type', type=str, default='regression')
    parser.add_argument('--measure_name', type=str, default='measure')

    parser.add_argument('--input_type', type=str, default='both', choices=['smiles', 'inchi', 'both'])
    parser.add_argument('--dataset_name', type=str, default='metlin')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--metric', type=str, default='loss')
    args = parser.parse_args()

    pl.seed_everything(42)
    args.measure_name = get_default_tasks(args.dataset_name)
    config = Config()
    s_tokenizer = ChemTokenizer(config.smiles_vocab_path, mode='smiles')
    i_tokenizer = ChemTokenizer(config.inchi_vocab_path, mode='inchi')

    def load_df(s):
        df = pd.read_csv(os.path.join(args.data_root, f"{args.dataset_name}_{s}.csv"))
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df

    train_df, valid_df = load_df('train'), load_df('valid')
    t_cols = [str(c).lower() for c in args.measure_name]
    target_mean = train_df[t_cols].apply(pd.to_numeric, errors='coerce').mean().values
    target_std = train_df[t_cols].apply(pd.to_numeric, errors='coerce').std().values
    target_std[target_std == 0] = 1.0

    train_ds = FusionDataset(train_df, s_tokenizer, i_tokenizer, 'regression', args.measure_name, args.input_type,
                             'TRAIN', target_mean, target_std)
    valid_ds = FusionDataset(valid_df, s_tokenizer, i_tokenizer, 'regression', args.measure_name, args.input_type,
                             'VALID', target_mean, target_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_ds.collate_fn, num_workers=4, drop_last=True)

    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=valid_ds.collate_fn, num_workers=4)

    model = FusionFinetuneModel(args, config, s_tokenizer, i_tokenizer, target_mean, target_std)

    mode = 'max' if args.metric in ['r2'] else 'min'
    monitor = f'val_{args.metric}' if args.metric != 'loss' else 'val_loss'

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=monitor, dirpath=f"checkpoints/{args.dataset_name}",
                                                       filename="best-{epoch:02d}", save_top_k=1, mode=mode)
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpu,
                         callbacks=[TQDMProgressBar(), checkpoint_callback, LearningRateMonitor()],
                         logger=pl.loggers.CSVLogger("logs", name=args.dataset_name))

    trainer.fit(model, train_loader, valid_loader)

    print(f"\n[-] Training Finished.")

    best_score = 0.0
    best_model_path = "No_Model_Saved"

    if checkpoint_callback.best_model_score is not None:
        best_score = float(checkpoint_callback.best_model_score.cpu().item())
        best_model_path = checkpoint_callback.best_model_path
        print(f"[-] Best Metric ({monitor}): {best_score:.4f}")
        print(f"[-] Best Model saved at: {best_model_path}")
    else:
        print(f"[!] Warning: Metric {monitor} not found or model not saved.")

    log_dir = trainer.logger.log_dir if trainer.logger else "No_Log"
    best_metrics = trainer.callback_metrics
    os.makedirs("result", exist_ok=True)
    summary_file = f"result/{args.dataset_name}_inchi2026_results.csv"
    file_exists = os.path.exists(summary_file)

    headers = [
        'Dataset', 'Input_Type', 'Task', 'Target',
        'Best_Model_Path', 'Metric_Name', 'Best_Score',
        'LR', 'Dropout', 'BatchSize', 'HiddenDim', 'NumLayers',
        'Log_Dir',
        'G_R2', 'G_RMSE', 'G_MedARE',
        'MH_R2', 'MH_RMSE', 'MH_MedARE',
        'M_H_R2', 'M_H_RMSE', 'M_H_MedARE',
        'MNa_R2', 'MNa_RMSE', 'MNa_MedARE'
    ]

    row_data = [
        args.dataset_name, args.input_type, getattr(args, 'task_type', 'regression'),
        str(args.measure_name)[:50],
        best_model_path,
        args.metric, f"{best_score:.4f}",
        args.learning_rate, args.dropout, args.batch_size, args.hidden_dim, args.num_layers,
        log_dir,
        f"{best_metrics.get('val_r2', 0):.4f}", f"{best_metrics.get('val_rmse', 0):.4f}",
        f"{best_metrics.get('val_medare', 0):.4f}",
        f"{best_metrics.get('val_MH_r2', 0):.4f}", f"{best_metrics.get('val_MH_rmse', 0):.4f}",
        f"{best_metrics.get('val_MH_medare', 0):.4f}",
        f"{best_metrics.get('val_M_H_r2', 0):.4f}", f"{best_metrics.get('val_M_H_rmse', 0):.4f}",
        f"{best_metrics.get('val_M_H_medare', 0):.4f}",
        f"{best_metrics.get('val_MNa_r2', 0):.4f}", f"{best_metrics.get('val_MNa_rmse', 0):.4f}",
        f"{best_metrics.get('val_MNa_medare', 0):.4f}"
    ]

    with open(summary_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row_data)

    print(f"[-] Summary and Hyperparameters saved to {summary_file}")


if __name__ == '__main__':
    main()