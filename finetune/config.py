class Config:
    def __init__(self):
        # ... (之前的路径配置保持不变) ...
        self.train_file = '/share/home/xiabing/ssc_jobs/MOLDE/data/train.h5'
        self.val_file = '/share/home/xiabing/ssc_jobs/MOLDE/data/val.h5'
        self.smiles_vocab_path = '/home/xiooli/sscjobs/MOLDE/data/static_tokenizer_smiles.txt'
        self.inchi_vocab_path = '/home/xiooli/sscjobs/MOLDE/data/static_tokenizer_inchi.txt'
        self.save_dir = './lstm_cnn'

        self.inchi_padding_idx = 0
        self.smiles_padding_idx = 0

        # ... (数据加载参数保持不变) ...
        self.batch_size = 512
        #384
        self.num_workers = 16
        self.max_inchi_len = 1024
        self.max_smiles_len = 1024

        # ---------------------------------------------------------------------
        # 模型参数 (LSTM + CNN Hybrid)
        # ---------------------------------------------------------------------

        # --- LSTM 参数 ---
        self.inchi_embed_dim = 768
        self.inchi_num_layers = 5

        self.smiles_embed_dim = 768
        self.smiles_num_layers = 5

        # --- CNN 参数 (新增) ---
        # 这里的 [3, 5, 7] 对应化学中的不同局部环境：
        # 3: 原子及其直接邻居 (Local)
        # 5: 扩展的官能团 (Intermediate)
        # 7: 较长的链或环系统 (Global/Regional)
        self.cnn_filters = 64  # 每个卷积核的通道数
        self.cnn_kernels = [3, 5, 7, 9]  # 卷积核尺寸列表
        self.cnn_num_layers = 4

        # --- 通用 ---
        self.dropout = 0.1
        self.projection_dim = 512

        # ... (训练参数保持不变) ...
        self.epochs = 200
        self.lr = 1e-5
        self.temperature = 0.07
        self.weight_decay = 1e-4
        self.device = 'cuda'
        self.grad_clip = 5.0
