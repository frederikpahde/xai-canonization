from canonizers.efficientnet import (EfficientNetBNCanonizer, EfficientNetCanonizer)

from canonizers.densenet import SequentialThreshCanonizer, ThreshSequentialCanonizer, DenseNetAdaptiveAvgPoolCanonizer
from canonizers.densenet_untangle import UntanglingCanonizer
from canonizers.resnet import ResNetCanonizer, ResNetBNCanonizer
from zennit.torchvision import VGGCanonizer
from canonizers.rn_canonizer import (RelationNetCanonizer,
                                                        RelationNetBNConvCanonizer, 
                                                        RelationNetBNAllCanonizer, 
                                                        RelationNetBNOnlyCanonizer)


from utils.helpers_quantus import wrap_zennit_quantus


def explainer_zennit_default(model, inputs, targets, device, *args, **kwargs):
    canonizer = None
    return wrap_zennit_quantus(canonizer, model, inputs, targets, device, *args, **kwargs)

# EfficientNet
def explainer_zennit_efficientnet(model, inputs, targets, device, *args, **kwargs):
    canonizer = EfficientNetCanonizer
    return wrap_zennit_quantus(canonizer, model, inputs, targets, device, *args, **kwargs)

def explainer_zennit_efficientnet_canonized(model, inputs, targets, device, *args, **kwargs):
    canonizer = EfficientNetBNCanonizer
    return wrap_zennit_quantus(canonizer, model, inputs, targets, device, *args, **kwargs)

# VGG
def explainer_zennit_bn_canonized(model, inputs, targets, device, *args, **kwargs):
    canonizer = VGGCanonizer
    return wrap_zennit_quantus(canonizer, model, inputs, targets, device, *args, **kwargs)

# ResNet
def explainer_zennit_resnet(model, inputs, targets, device, *args, **kwargs):
    canonizer = ResNetCanonizer
    return wrap_zennit_quantus(canonizer, model, inputs, targets, device, *args, **kwargs)

def explainer_zennit_resnet_canonized(model, inputs, targets, device, *args, **kwargs):
    canonizer = ResNetBNCanonizer
    return wrap_zennit_quantus(canonizer, model, inputs, targets, device, *args, **kwargs)

# Relation Network
def explainer_zennit_rn(model, inputs, targets, device, *args, **kwargs):
    canonizer = RelationNetCanonizer
    return wrap_zennit_quantus(canonizer, model, inputs, targets, device, *args, **kwargs)

def explainer_zennit_rn_conv_canonized(model, inputs, targets, device, *args, **kwargs):
    canonizer = RelationNetBNConvCanonizer
    return wrap_zennit_quantus(canonizer, model, inputs, targets, device, *args, **kwargs)

def explainer_zennit_rn_all_canonized(model, inputs, targets, device, *args, **kwargs):
    canonizer = RelationNetBNAllCanonizer
    return wrap_zennit_quantus(canonizer, model, inputs, targets, device, *args, **kwargs)

def explainer_zennit_rn_bn_only_canonized(model, inputs, targets, device, *args, **kwargs):
    canonizer = RelationNetBNOnlyCanonizer
    return wrap_zennit_quantus(canonizer, model, inputs, targets, device, *args, **kwargs)

# DenseNet
def explainer_zennit_densenet(model, inputs, targets, device, *args, **kwargs):
    return wrap_zennit_quantus(DenseNetAdaptiveAvgPoolCanonizer, model, inputs, targets, device, *args, **kwargs)

def explainer_zennit_densenet_canonized_seq_thresh(model, inputs, targets, device, *args, **kwargs):
    return wrap_zennit_quantus(SequentialThreshCanonizer, model, inputs, targets, device, *args, **kwargs)

def explainer_zennit_densenet_canonized_thresh_seq(model, inputs, targets, device, *args, **kwargs):
    return wrap_zennit_quantus(ThreshSequentialCanonizer, model, inputs, targets, device, *args, **kwargs)

def explainer_zennit_densenet_untangled(model, inputs, targets, device, *args, **kwargs):
    return wrap_zennit_quantus(UntanglingCanonizer, model, inputs, targets, device, *args, **kwargs)

