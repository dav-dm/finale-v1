from torch import nn
from torch.autograd import grad


def clone_module(module, memo=None):
    """
    Clones a module preserving the computational graph, so gradients
    back-propagate through the clone all the way to the original parameters.
    memo deduplicates shared parameters across submodules.
    """
    if memo is None:
        memo = {}
    if not isinstance(module, nn.Module):
        return module

    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    for key, param in module._parameters.items():
        if param is None:
            continue
        ptr = param.data_ptr
        if ptr in memo:
            clone._parameters[key] = memo[ptr]
        else:
            cloned = param.clone()
            clone._parameters[key] = cloned
            memo[ptr] = cloned

    for key, buf in module._buffers.items():
        if buf is None or not buf.requires_grad:
            continue
        ptr = buf.data_ptr
        if ptr in memo:
            clone._buffers[key] = memo[ptr]
        else:
            cloned = buf.clone()
            clone._buffers[key] = cloned
            memo[ptr] = cloned

    for key in clone._modules:
        clone._modules[key] = clone_module(module._modules[key], memo=memo)

    # Rebuild flattened parameters for RNNs
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)

    return clone


def update_module(module, updates=None, memo=None):
    """
    Updates module parameters in-place as p ← p + update, preserving
    differentiability (no in-place tensor ops - parameters are rerouted).
    If updates is provided, assigns each element to the corresponding p.update
    before applying. memo deduplicates shared parameters.
    """
    if memo is None:
        memo = {}

    if updates is not None:
        params = list(module.parameters())
        if len(updates) != len(params):
            print(
                f'WARNING update_module(): parameters and updates have different '
                f'lengths ({len(params)} vs {len(updates)})'
            )
        for p, u in zip(params, updates):
            p.update = u

    for key, p in module._parameters.items():
        if p is None or not hasattr(p, 'update') or p.update is None:
            continue
        if p in memo:
            module._parameters[key] = memo[p]
        else:
            updated = p + p.update
            p.update = None
            memo[p] = updated
            module._parameters[key] = updated

    for key, buf in module._buffers.items():
        if buf is None or not hasattr(buf, 'update') or buf.update is None:
            continue
        if buf in memo:
            module._buffers[key] = memo[buf]
        else:
            updated = buf + buf.update
            buf.update = None
            memo[buf] = updated
            module._buffers[key] = updated

    for key in module._modules:
        module._modules[key] = update_module(module._modules[key], memo=memo)

    # Rebuild flattened parameters for RNNs
    if hasattr(module, 'flatten_parameters'):
        module._apply(lambda x: x)

    return module


class MAMLWrapper(nn.Module):
    """
    [[Link to Source Code]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)
    MAML wrapper providing clone() and adapt() for inner-loop adaptation.
    The outer-loop model is never modified directly, adaptation always happens on a
    clone(), so meta-learned parameters stay intact between episodes.

    """
    def __init__(self, model, lr, first_order=False, allow_unused=False, allow_nograd=False):
        super().__init__()
        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.allow_nograd = allow_nograd
        self.allow_unused = allow_unused or allow_nograd

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self, loss, first_order=None, allow_unused=None, allow_nograd=None):
        """
        Computes inner-loop gradients and updates cloned parameters via tensor
        rerouting (no in-place ops), preserving the compute graph for second-order MAML.
        """
        first_order = self.first_order if first_order is None else first_order
        allow_unused = self.allow_unused if allow_unused is None else allow_unused
        allow_nograd = self.allow_nograd if allow_nograd is None else allow_nograd
        second_order = not first_order

        if allow_nograd:
            diff_params = [p for p in self.module.parameters() if p.requires_grad]
            diff_grads = grad(
                loss, diff_params,
                retain_graph=second_order,
                create_graph=second_order,
                allow_unused=allow_unused,
            )
            diff_iter = iter(diff_grads)
            gradients = [
                next(diff_iter) if p.requires_grad else None
                for p in self.module.parameters()
            ]
        else:     
            gradients = grad(
                loss, self.module.parameters(),
                retain_graph=second_order,
                create_graph=second_order,
                allow_unused=allow_unused,
            )

        for p, g in zip(self.module.parameters(), gradients):
            if g is not None:
                p.update = -self.lr * g
        self.module = update_module(self.module)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
        Returns a MAML-wrapped deep clone whose gradients propagate back
        to the original meta-learned parameters.
        """
        return MAMLWrapper(
            clone_module(self.module),
            lr=self.lr,
            first_order=self.first_order if first_order is None else first_order,
            allow_unused=self.allow_unused if allow_unused is None else allow_unused,
            allow_nograd=self.allow_nograd if allow_nograd is None else allow_nograd,
        )