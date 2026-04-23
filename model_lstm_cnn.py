import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class ResidualConv1d(nn.Module):


    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super(ResidualConv1d, self).__init__()
        # padding=kernel_size//2 保证输入输出序列长度一致（假设 stride=1）
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + residual


class HybridMoleculeEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, dropout,
                 cnn_filters=64, cnn_kernels=[3, 5, 7], cnn_num_layers=1):
        super(HybridMoleculeEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout_layer = nn.Dropout(dropout)


        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )


        self.cnn_kernels = cnn_kernels
        self.stem_convs = nn.ModuleList([
            nn.Conv1d(in_channels=d_model,
                      out_channels=cnn_filters,
                      kernel_size=k,
                      padding=k // 2)
            for k in cnn_kernels
        ])

        self.cnn_feature_dim = cnn_filters * len(cnn_kernels)

        self.deep_cnn_layers = nn.ModuleList()
        for _ in range(cnn_num_layers - 1):
            self.deep_cnn_layers.append(
                ResidualConv1d(channels=self.cnn_feature_dim,
                               kernel_size=3,  # 中间层通常用小核堆叠
                               dropout=dropout)
            )

        self.cnn_project = nn.Sequential(
            nn.Linear(self.cnn_feature_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, src, padding_idx=0):

        src = torch.clamp(src, min=0, max=self.vocab_size - 1)

        x = self.embedding(src)
        x = self.dropout_layer(x)  # [Batch, Seq, Dim]


        lengths = (src != padding_idx).sum(dim=1).cpu()
        lengths = torch.clamp(lengths, min=1)
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_x)
        lstm_feature = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=-1)

        x_permuted = x.permute(0, 2, 1)

        stem_outputs = []
        for conv in self.stem_convs:
            out = F.relu(conv(x_permuted))
            stem_outputs.append(out)

        cnn_feat = torch.cat(stem_outputs, dim=1)

        for res_layer in self.deep_cnn_layers:
            cnn_feat = res_layer(cnn_feat)

        if cnn_feat.size(2) > 1:
            cnn_feat = F.max_pool1d(cnn_feat, kernel_size=cnn_feat.size(2)).squeeze(2)
        else:
            cnn_feat = cnn_feat.squeeze(2)


        cnn_feature_proj = self.cnn_project(cnn_feat)

        gate_input = torch.cat([lstm_feature, cnn_feature_proj], dim=1)
        alpha = self.gate_net(gate_input)
        fused_vector = alpha * lstm_feature + (1 - alpha) * cnn_feature_proj
        final_vector = self.final_norm(fused_vector)

        return final_vector


class DualEncoderModel(nn.Module):
    def __init__(self, inchi_vocab_size, smiles_vocab_size, config):
        super(DualEncoderModel, self).__init__()

        self.inchi_pad_idx = getattr(config, 'inchi_padding_idx', 0)
        self.smiles_pad_idx = getattr(config, 'smiles_padding_idx', 0)

        cnn_filters = getattr(config, 'cnn_filters', 64)
        cnn_kernels = getattr(config, 'cnn_kernels', [3, 5, 7])
        cnn_num_layers = getattr(config, 'cnn_num_layers', 1)

        # --- InChI Encoder ---
        self.inchi_encoder = HybridMoleculeEncoder(
            vocab_size=inchi_vocab_size,
            d_model=config.inchi_embed_dim,
            num_layers=config.inchi_num_layers,
            dropout=config.dropout,
            cnn_filters=cnn_filters,
            cnn_kernels=cnn_kernels,
            cnn_num_layers=cnn_num_layers
        )

        # --- SMILES Encoder ---
        self.smiles_encoder = HybridMoleculeEncoder(
            vocab_size=smiles_vocab_size,
            d_model=config.smiles_embed_dim,
            num_layers=config.smiles_num_layers,
            dropout=config.dropout,
            cnn_filters=cnn_filters,
            cnn_kernels=cnn_kernels,
            cnn_num_layers=cnn_num_layers
        )

        # Projection Heads (保持不变)
        self.inchi_proj = nn.Sequential(
            nn.Linear(config.inchi_embed_dim, config.inchi_embed_dim),
            nn.BatchNorm1d(config.inchi_embed_dim),
            nn.ReLU(),
            nn.Linear(config.inchi_embed_dim, config.projection_dim),
            nn.BatchNorm1d(config.projection_dim)
        )

        self.smiles_proj = nn.Sequential(
            nn.Linear(config.smiles_embed_dim, config.smiles_embed_dim),
            nn.BatchNorm1d(config.smiles_embed_dim),
            nn.ReLU(),
            nn.Linear(config.smiles_embed_dim, config.projection_dim),
            nn.BatchNorm1d(config.projection_dim)
        )

    def forward(self, inchi_ids, smiles_ids):
        inchi_feat = self.inchi_encoder(inchi_ids, padding_idx=self.inchi_pad_idx)
        inchi_out = self.inchi_proj(inchi_feat)

        smiles_feat = self.smiles_encoder(smiles_ids, padding_idx=self.smiles_pad_idx)
        smiles_out = self.smiles_proj(smiles_feat)

        inchi_out = F.normalize(inchi_out, p=2, dim=1)
        smiles_out = F.normalize(smiles_out, p=2, dim=1)

        return inchi_out, smiles_out

