import torch
import torchvision
import logging
from torch.nn.modules import Sequential, Linear
from collections import OrderedDict
from models.relation_network import RelationNet

logger = logging.getLogger(__name__)


def load_model(dataset_name, model_name, model_path=None):
    """Load (pre-trained) models.

    Args:
        model_name (str): name of model (VGG/ResNet/EfficientNet)
        efficientnet_implementation (str, optional): source of pre-trained efficientnet model (timm/lukemelas). Defaults to "timm".

    Returns:
        torch.nn.Module: model to be evaluated
    """
    model=None
    if dataset_name == "imagenet":
        if model_name == "efficientnet_b0":
            return torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights)
        elif model_name == "efficientnet_b4":
            return torchvision.models.efficientnet_b4(weights=torchvision.models.EfficientNet_B4_Weights)
        elif model_name == "resnet18":
            model = torchvision.models.resnet18(pretrained=True)
        elif model_name == "resnet50":
            model = torchvision.models.resnet50(pretrained=True)
        elif model_name == "vgg16":
            model = torchvision.models.vgg16_bn(pretrained=True)
        elif "densenet" in model_name:
            if model_name == "densenet_121":
                model = torchvision.models.densenet121(pretrained=True, memory_efficient=False)
            elif model_name == "densenet_161":
                model = torchvision.models.densenet161(pretrained=True, memory_efficient=False)
            elif model_name == "densenet_169":
                model = torchvision.models.densenet169(pretrained=True, memory_efficient=False)
            elif model_name == "densenet_201":
                model = torchvision.models.densenet201(pretrained=True, memory_efficient=False)
    elif dataset_name == "VOC":
        model = load_VOC_model(model_name, model_path)
    elif dataset_name == "MS":
        model = load_MS_model(model_name, model_path)

    if model_name == "relation_network":
        model = load_relation_network(model_path)

    if model is None:
        raise ValueError(f"Unknown model name {model_name} for dataset {dataset_name}")
    # Setting 'inplace' to False for SiLU/ReLU Layers
    for i, (name, layer) in enumerate(model.named_modules()):
        if type(layer) in [torch.nn.modules.activation.ReLU,
                           torch.nn.modules.activation.SiLU]:
            layer.inplace = False
    return model.eval()

def load_VOC_model(model_name, model_path):
    logger.info(f"Loading VOC model from {model_path}")
    model = load_model("imagenet",model_name)
    if "efficientnet" in model_name or "vgg" in model_name:
        in_features = model.classifier[-1].in_features
        model.classifier = Sequential(*(list(model.classifier[:-1]) + [Linear(in_features, 20)]))
    elif "resnet" in model_name:
        model.fc = Linear(model.fc.in_features, 20)
    elif "densenet" in model_name:
        model.classifier = Linear(model.classifier.in_features, 20)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    return model

def load_MS_model(model_name, model_path):
    logger.info(f"Loading MS model from {model_path}")
    model = load_model("imagenet",model_name)
    if "efficientnet" in model_name or "vgg" in model_name:
        in_features = model.classifier[-1].in_features
        model.classifier = Sequential(*(list(model.classifier[:-1]) + [Linear(in_features, 80)]))
    elif "resnet" in model_name:
        model.fc = Linear(model.fc.in_features, 80)
    elif "densenet" in model_name:
        model.classifier = Linear(model.classifier.in_features, 80)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    return model

def load_relation_network(model_path):
    net = RelationNet()
    state_dict = torch.load(model_path)

    key_map = {'conv1': 'image_encoder.0',
           'norm1': 'image_encoder.2',
           'conv2': 'image_encoder.3',
           'norm2': 'image_encoder.5',
           'conv3': 'image_encoder.6',
           'norm3': 'image_encoder.8',
           'conv4': 'image_encoder.9',
           'norm4': 'image_encoder.11'}

    def overwrite_name(name, key_map):
        for old_name, new_name in key_map.items():
            name = name.replace(old_name, new_name)
        return name

    state_dict_new = OrderedDict({overwrite_name(key, key_map): item for key, item in state_dict.items()})

    net.load_state_dict(state_dict_new)
    net = net.eval()
    return net

def load_relation_network_on_device(model_path, device):
    net = RelationNet(device=device)
    state_dict = torch.load(model_path, map_location=device)

    key_map = {'conv1': 'image_encoder.0',
           'norm1': 'image_encoder.2',
           'conv2': 'image_encoder.3',
           'norm2': 'image_encoder.5',
           'conv3': 'image_encoder.6',
           'norm3': 'image_encoder.8',
           'conv4': 'image_encoder.9',
           'norm4': 'image_encoder.11'}

    def overwrite_name(name, key_map):
        for old_name, new_name in key_map.items():
            name = name.replace(old_name, new_name)
        return name

    state_dict_new = OrderedDict({overwrite_name(key, key_map): item for key, item in state_dict.items()})
    net.load_state_dict(state_dict_new)
    net = net.to(device).eval()
    return net