from zennit.composites import SpecialFirstLayerMapComposite
from zennit.rules import ZBox, ZPlus, Pass, Norm
from zennit.types import Convolution, Activation, AvgPool, Linear, BatchNorm
from zennit.layer import Sum


class ExcitationBackpropBox(SpecialFirstLayerMapComposite):
    def __init__(self, low=0.0, high=1.0, stabilizer=1e-6, layer_map=None, first_map=[], 
                 zero_params=None, canonizers=None, **rule_kwargs):
        if layer_map is None:
            layer_map = []

        layer_map = layer_map + [
            (Sum, Norm(stabilizer=stabilizer)),
            (Activation, Pass()),
            (BatchNorm, Pass()),
            (AvgPool, Norm(stabilizer=stabilizer)),
            (Linear, ZPlus(stabilizer=stabilizer, zero_params=zero_params)),
        ]
        
        first_map = first_map + [
            (Convolution, ZBox(low=low, high=high, stabilizer=stabilizer, **rule_kwargs))
        ]
        super().__init__(layer_map=layer_map, first_map=first_map, canonizers=canonizers)
