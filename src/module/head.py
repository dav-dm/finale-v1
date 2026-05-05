import math
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from qpth.qp import QPFunction


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
    

class DistLinear(nn.Module):
    """
    From "A Closer Look at Few-shot Classification. ICLR 2019."
    Implemented from: https://github.com/RL-VIG/LibFewShot
    """
    def __init__(self, in_channel, out_channel):
        super(DistLinear, self).__init__()
        self.fc = nn.Linear(in_channel, out_channel, bias=False)

        self.class_wise_learnable_norm = True
        if self.class_wise_learnable_norm:
            weight_norm(self.fc, name="weight", dim=0)

        self.scale_factor = 2 if out_channel <= 200 else 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)

        if not self.class_wise_learnable_norm:
            fc_norm = (
                torch.norm(self.fc.weight.data, p=2, dim=1)
                .unsqueeze(1)
                .expand_as(self.fc.weight.data)
            )
            self.fc.weight.data = self.fc.weight.data.div(fc_norm + 0.00001)

        cos_dist = self.fc(x_normalized)
        score = self.scale_factor * cos_dist

        return score
    

class NegativeMarginLayer(nn.Module):
    """
    From "Negative Margin Matters: Understanding Margin in Few-shot Classification"
    Implemented from: https://github.com/RL-VIG/LibFewShot
    """
    def __init__(self, in_features, out_features, margin=-0.3, temperature=30.0):
        super(NegativeMarginLayer, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        device = feature.device
        cosine = nn.functional.linear(
            nn.functional.normalize(feature), 
            nn.functional.normalize(self.weight.to(device))
        )
        # when test, no label, just return
        if label is None:
            return cosine * self.temperature

        phi = cosine - self.margin

        output = torch.where(self.one_hot(label, cosine.shape[1]).bool(), phi, cosine)
        output *= self.temperature
        return output

    def one_hot(self, y, num_class):
        return (
            torch.zeros((len(y), num_class)).to(y.device).scatter_(1, y.unsqueeze(1), 1)
        )


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
    

class PrototypicalHead(nn.Module):
    """
    Prototype-based head for Prototypical Networks.
    Classifies queries by negative squared Euclidean distance to class prototypes.
    """
    def __init__(self, distance='euclidean', normalize=False, eps=1e-8):
        super().__init__()
        if distance not in ('euclidean', 'cosine'):
            raise ValueError("distance must be 'euclidean' or 'cosine'")
        
        self.distance = distance
        self.normalize = normalize or (distance == 'cosine')
        self.eps = eps

        self.register_buffer("prototypes", None, persistent=False)
        self.register_buffer("proto_classes", None, persistent=False)

    def fit(self, x, y):
        if self.normalize:
            x = nn.functional.normalize(x, p=2, dim=1)
            
        classes = torch.unique(y, sorted=True)
        self.proto_classes = classes
        self.prototypes = torch.stack([
            x[y == cls].mean(dim=0) for cls in classes
        ])  # [num_classes, emb_dim]

    def forward(self, x):
        if self.prototypes is None:
            return torch.empty((x.size(0), 0), device=x.device)
        
        if self.normalize:
            x = nn.functional.normalize(x, p=2, dim=1)

        if self.distance == 'euclidean':
            dists = torch.cdist(x, self.prototypes).pow(2)
            return -dists  # [num_query, num_classes]
        else:  # cosine, both already normalized
            return x @ self.prototypes.T
        

class SVClassifier(nn.Module):
    """
    Differentiable SVM classifier used by MetaOptNet.
    Source code: learn2learn.nn.SVClassifier.
    """
    def __init__(
        self, support=None, labels=None, ways=None, normalize=False, C_reg=0.1, max_iters=15
    ):
        super().__init__()

        self.C_reg = C_reg
        self.max_iters = max_iters
        self._normalize = normalize

        self.num_support = None
        self.ways = None

        self.register_buffer("support", None, persistent=False)
        self.register_buffer("qp_solution", None, persistent=False)

        if support is not None and labels is not None:
            if ways is None:
                ways = int(labels.max().item()) + 1
            self.fit(support, labels, ways=ways)

    def fit(self, support, labels, ways=None, C_reg=None, max_iters=None):
        """
        Fits the differentiable SVM on support embeddings.

        Steps:
            kernel = support @ support.T
            block_kernel = kronecker(kernel, I_ways) + I
            labels_onehot = onehot(labels, ways).view(1, -1)
            h = C_reg * labels_onehot
            A = kronecker(I_support, ones(1, ways))
            b = zeros(1, num_support)

        The QP solution is then stored and reused in forward().
        """
        if C_reg is None:
            C_reg = self.C_reg

        if max_iters is None:
            max_iters = self.max_iters

        labels = labels.long()

        if self._normalize:
            support = self.normalize(support)

        if ways is None:
            ways = int(labels.max().item()) + 1

        ways = int(ways)

        num_support = support.size(0)
        device = support.device
        dtype = support.dtype

        # Linear kernel matrix over support samples.
        kernel = support @ support.t()

        # Identity matrix in class space.
        I_ways = torch.eye(ways, device=device, dtype=dtype)

        # Build block kernel (K \otimes I) and add diagonal stabilization term.
        block_kernel = self.kronecker(kernel, I_ways)
        block_kernel = block_kernel + torch.eye(
            ways * num_support, device=device, dtype=dtype,
        )

        # Flatten one-hot labels to match QP vectorized formulation.
        labels_onehot = self.onehot(labels, dim=ways, dtype=dtype).view(1, -1).to(device)

        # Identity matrices for inequality and equality constraint construction.
        I_sw = torch.eye(num_support * ways, device=device, dtype=dtype)
        I_s = torch.eye(num_support, device=device, dtype=dtype)

        # Upper bound vector for box constraints.
        h = C_reg * labels_onehot

        # Equality constraints: sum of class coefficients per support sample.
        A = self.kronecker(
            I_s,
            torch.ones(1, ways, device=device, dtype=dtype,),
        )

        # Right-hand side of equality constraints.
        b = torch.zeros(1, num_support, device=device, dtype=dtype)

        # Differentiable QP solver instance.
        qp = QPFunction(verbose=False, maxIter=max_iters)

        # Solve the QP to obtain dual variables.
        qp_solution = qp(block_kernel, -labels_onehot, I_sw, h, A, b)

        self.qp_solution = qp_solution.reshape(num_support, ways)
        self.support = support
        self.num_support = num_support
        self.ways = ways

        return self.qp_solution

    def forward(self, x):
        """
        Computes query logits using the fitted SVM solution.
        """
        if self.support is None or self.qp_solution is None:
            return torch.empty((x.size(0), 0), device=x.device)

        if self._normalize:
            x = self.normalize(x)

        num_query = x.size(0)

        # Expand the stored QP solution to align with support x query x classes.
        # qp_solution: [num_support, num_query, ways]
        qp_solution = self.qp_solution.unsqueeze(1).expand(
            self.num_support, num_query, self.ways,
        )
        # Compute similarity (compatibility) between supports and queries.
        # support: [num_support, feature_dim], x.t(): [feature_dim, num_query]
        # compatibility: [num_support, num_query]
        compatibility = self.support @ x.t()

        # Expand compatibility to match the class dimension: [num_support, num_query, ways]
        compatibility = compatibility.unsqueeze(2).expand(
            self.num_support, num_query, self.ways,
        )

        # Element-wise multiply contributions from each support and sum over supports
        # to obtain final logits per query and class: [num_query, ways]
        logits = qp_solution * compatibility
        return torch.sum(logits, dim=0)    
    
    @staticmethod
    def normalize(x, epsilon=1e-8):
        return x / (x.norm(p=2, dim=1, keepdim=True) + epsilon)

    @staticmethod
    def kronecker(A, B):
        """
        A: [a1, a2]
        B: [b1, b2]

        returns: [a1 * b1, a2 * b2]
        """
        return torch.einsum("ab,cd->acbd", A, B).view(
            A.size(0) * B.size(0),
            A.size(1) * B.size(1),
        )

    @staticmethod
    def onehot(x, dim, dtype=torch.float32):
        """
        Same logic as learn2learn onehot().
        """
        size = x.size(0)
        x = x.long()
        out = torch.zeros(size, dim, device=x.device, dtype=dtype)
        out.scatter_(1, x.view(-1, 1), 1.0)
        return out