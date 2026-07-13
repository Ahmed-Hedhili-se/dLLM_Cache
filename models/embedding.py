import torch.nn as nn
from models.utils import VS, H

class TokenEmbedding(nn.Embedding):

    def __init__(self):
        super().__init__(VS, H)