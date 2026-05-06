class Config:
    def __init__(self):
        self.train_file = 'data/processed_data/METLIN_train.csv'
        self.val_file = 'data/processed_data/METLIN_test.csv'
        self.test_file = 'data/processed_data/METLIN_test.csv'

        self.ecfp_num = 1024
        self.adduct_num = 3
        self.inchi_vocab_path = '/share/home/xiabing/ssc_jobs/MOLDE/data/static_tokenizer_inchi.txt'
        self.smiles_vocab_path = '/share/home/xiabing/ssc_jobs/MOLDE/data/static_tokenizer_smiles.txt'
        self.save_dir = './lstm_cnn'

        self.inchi_padding_idx = 0
        self.smiles_padding_idx = 0

        self.batch_size = 512
        self.num_workers = 16
        self.max_inchi_len = 1024
        self.max_smiles_len = 1024

        self.inchi_embed_dim = 768
        self.inchi_num_layers = 5

        self.smiles_embed_dim = 768
        self.smiles_num_layers = 5


        self.cnn_filters = 64
        self.cnn_kernels = [3, 5, 7, 9]
        self.cnn_num_layers = 4

        self.dropout = 0.1
        self.projection_dim = 512

        self.epochs = 200
        self.lr = 1e-5
        self.temperature = 0.07
        self.weight_decay = 1e-4
        self.device = 'cuda'
        self.grad_clip = 5.0
