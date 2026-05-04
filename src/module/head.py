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


class MatchingHead(nn.Module):
    """
    Attention-based Matching Networks head.
    """
    def __init__(self, normalize=True, eps=1e-8):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

        self.register_buffer("support", None, persistent=False)
        self.register_buffer("support_y", None, persistent=False)

    def fit(self, x, y):
        if self.normalize:
            x = nn.functional.normalize(x, p=2, dim=1)

        self.support = x
        self.support_y = y

    def forward(self, x):
        if self.support is None or self.support_y is None:
            return torch.empty((x.size(0), 0), device=x.device)

        if self.normalize:
            x = nn.functional.normalize(x, p=2, dim=1)

        # Similarity query-support: [num_query, num_support]
        scores = x @ self.support.T

        # Attention over support samples
        attn = torch.softmax(scores, dim=1)

        num_classes = int(self.support_y.max().item()) + 1

        # One-hot encode support labels: [num_support, num_classes]
        one_hot = nn.functional.one_hot(
            self.support_y,
            num_classes=num_classes,
        ).to(dtype=attn.dtype)

        # Class probabilities: [num_query, num_classes]
        probs = attn @ one_hot

        return torch.log(probs.clamp_min(self.eps))