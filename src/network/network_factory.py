import importlib

from util.config import load_config

networks = {
    '2dcnn' : 'TwoDCNN',
    'transformer' : 'BiflowTransformer',
    'gru' : 'GRU',
    'unet' : 'UNet',
}


def build_network(**kwargs):
    """
    Instantiates and returns a neural network model corresponding
    to the given string identifier.
    """
    cf = load_config()
    network_name = kwargs.get('network', cf['network'])
    Network = getattr(importlib.import_module(
        name=f'network.{network_name}'), networks[network_name])
    return Network(**kwargs)