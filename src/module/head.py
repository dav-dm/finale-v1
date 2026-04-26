import torch
import torch.nn as nn


class MultiHead(nn.Module):
    def __init__(self, heads: dict):
        super().__init__()
        self.heads = nn.ModuleDict(heads)

    def forward(self, x):
        return {name: head(x) for name, head in self.heads.items()}
    

class FullyConnected(nn.Module):
    """
    Fully-connected (linear) head
    """
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.fc(x)


class NNHead(nn.Module):
    """
    Optimized soft 1-nearest-neighbor head.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('embeddings_db', None)
        self.register_buffer('labels_db', None)
               
    def fit(self, x, y):
        # Memorize the train samples
        x = nn.functional.normalize(x, p=2, dim=1)
        self.embeddings_db = x
        self.labels_db = y

    def forward(self, x):
        # L2 normalization for Euclidean distance
        x = nn.functional.normalize(x, p=2, dim=1)
        
        # Compute all pairwise distances between queries and support samples
        distance_matrix = torch.cdist(x, self.embeddings_db)  # [x, self.embeddings_db]
        
        # Get unique classes and initialize storage
        classes = torch.unique(self.labels_db, sorted=True)
        num_classes = classes.size(0)
        query_size = x.size(0)
        
        # Find minimum distance for each class using vectorized operations
        min_distances = torch.empty(
            (query_size, num_classes), 
            device=x.device, 
            dtype=distance_matrix.dtype
        )
        
        for i, cls in enumerate(classes):
            class_mask = (self.labels_db == cls)
            if class_mask.any():
                min_distances[:, i] = distance_matrix[:, class_mask].min(dim=1).values
            else:
                min_distances[:, i] = torch.inf
                
        # Compute probabilities using softmax over negative distances
        logits = -min_distances
        # soft_values = torch.softmax(logits, dim=1)
        
        # # Generate predictions
        # idx = torch.argmax(soft_values, dim=1)
        # y_pred = classes[idx] 
        return logits
