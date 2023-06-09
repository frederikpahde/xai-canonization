from zennit import canonizers as zcanon
from zennit import torchvision as ztv

class ResNetCanonizer(zcanon.CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''
    def __init__(self):
        super().__init__((
            ztv.ResNetBottleneckCanonizer(),
            ztv.ResNetBasicBlockCanonizer(),
        ))

class ResNetBNCanonizer(zcanon.CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''
    def __init__(self):
        super().__init__((
            ztv.SequentialMergeBatchNorm(),
            ztv.ResNetBottleneckCanonizer(),
            ztv.ResNetBasicBlockCanonizer(),
        ))
