import torch
import torch.nn as nn

class AbstractRecommendationModel(nn.Module):
    
    def __init__(self, hparams: dict, word_embed_weights: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    
    