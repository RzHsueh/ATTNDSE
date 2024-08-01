import torch

from utils import *

class Config(object):
    def __init__(self):
        super().__init__()
        # 600.perlbench_s 623.xalancbmk_s 996.specrand_fs 602.gcc_s
        self.dataset = "600.perlbench_s" 
        self.seed = 42

        if torch.cuda.is_available():
            self.device = torch.device('cuda:1')
        else:
            self.device = torch.device('cpu')

        self.target = "ipc" # mode can be set as "ipc" or "power" or "area"
        self.moo = "cpi-power" # (multi_objective_optimization) mode can be "cpi-power" "cpi-area"

        self.embed_dim = 256
        self.num_heads = 8
        self.dropout = 0.1

        self.batch_size = 64
        self.lr = 3e-4
        self.depth = 3
        self.epochs = 200
        self.tolerence = 0.01

        self.position_encode = "sequence" # random sequence none
        self.shuffle_indices = generate_shuffle_indices(26)

    def set_from_parser(self, parser_instance):
        if hasattr(parser_instance, 'dataset'):
            self.dataset = parser_instance.dataset
        if hasattr(parser_instance, 'moo'):
            self.moo = parser_instance.moo


args = Config()