from torchvision.models.densenet import DenseNet, _DenseBlock, _Transition
from torch.nn import Sequential, Module, AvgPool2d, BatchNorm2d
import torch
from torch.nn.functional import relu as ReLU
from copy import deepcopy
from zennit.types import ConvolutionTranspose
from zennit.core import collect_leaves
from zennit.canonizers import AttributeCanonizer
import torchvision
from .densenet import CorrectSequentialMergeBatchNorm, CorrectCompositeCanonizer, ThreshReLUMergeBatchNorm, SequentialMergeBatchNormtoRight

class CanonizedDenseBlock(Module):
    @staticmethod
    def get_bn_params(bn, start, end):
        assert end>start
        return {
            "running_var": bn.running_var[start:end],
            "running_mean": bn.running_mean[start:end],
            "weight": bn.weight.data[start:end],
            "eps": bn.eps,
            "num_features": end - start,
            "bias": bn.bias.data[start:end]
        }

    @staticmethod
    def merge(conv, bn):
        denominator = (bn["running_var"] + bn["eps"]) ** .5
        scale = (bn["weight"] / denominator)

        original_weight = conv.weight.data
        if conv.bias is None:
            conv.bias = torch.nn.Parameter(
                torch.zeros(1, device=original_weight.device, dtype=original_weight.dtype)
            )
        original_bias = conv.bias.data

        if isinstance(conv, ConvolutionTranspose):
            index = (None, slice(None), *((None,) * (original_weight.ndim - 2)))
        else:
            index = (slice(None), *((None,) * (original_weight.ndim - 1)))

        # merge batch_norm into linear layer
        conv.weight.data = (original_weight * scale[index])
        conv.bias.data = (original_bias - bn["running_mean"]) * scale + bn["bias"]

    def __init__(self, block1, transition1=None, block2=None, transition2=None, final_norm=None, initial=False):
        assert (bool(transition2) <= bool(transition1))
        assert (bool(initial) <= bool(transition1))
        assert (bool(transition1) <= bool(block2))
        super().__init__()
        self.final = final_norm is not None
        self.len = len(block1)
        self.layers = [{} for _ in range(self.len)]
        self.initial = initial
        slice_init = block1["denselayer1"].norm1.num_features
        self.initial_bns = None
        self.initial_relus = None
        if self.initial: # If this is the first dense block
            self.initial_bns = []
            self.initial_relus = []
            for i in range(len(block1)):
                params = CanonizedDenseBlock.get_bn_params(block1[f"denselayer{i + 1}"].norm1, 0, slice_init)
                bn = torch.nn.BatchNorm2d(num_features=params['num_features'], eps=params['eps'])
                bn.running_mean = params['running_mean']
                bn.running_var = params['running_var']
                bn.weight.data = params['weight']
                bn.bias.data = params['bias']
                bn.eval()
                self.initial_bns.append(bn) #remember required parts of the initial bn layer parameters and relu objects
                relu = deepcopy(block1[f"denselayer{i + 1}"].relu1)
                self.initial_relus.append(relu)
            params=CanonizedDenseBlock.get_bn_params(transition1.norm,0,slice_init)
            bn = torch.nn.BatchNorm2d(num_features=params['num_features'], eps=params['eps'])
            bn.running_mean = params['running_mean']
            bn.running_var = params['running_var']
            bn.weight.data = params['weight']
            bn.bias.data = params['bias']
            bn.eval()
            self.initial_bns.append(bn) # and the transition layer
            relu=deepcopy(transition1.relu)
            self.initial_relus.append(relu)
        slice_init = block1.denselayer1.norm1.num_features

        for i, l in enumerate(self.layers):
            l['conv2_count']=0
            layer1 = block1[f"denselayer{i + 1}"]
            l["conv1"] = deepcopy(layer1.conv1)
            for j in range(i + 1, self.len + 1):
                l[f"conv2_{j}"] = deepcopy(layer1.conv2)
                l['conv2_count']=l['conv2_count']+1
                if j < self.len:
                    norm = deepcopy(block1[f"denselayer{j + 1}"].norm1)
                else:
                    if transition1 is not None:
                        norm = deepcopy(transition1.norm)
                    else:
                        norm = deepcopy(final_norm)
                bn_params = CanonizedDenseBlock.get_bn_params(norm, slice_init,
                                                              slice_init + layer1.conv2.out_channels)
                CanonizedDenseBlock.merge(l[f"conv2_{j}"], bn_params)
            slice_init = slice_init + layer1.conv2.out_channels
        if transition1 is not None: # if there is a transition layer following this denseblock
            self.transition = {}
            for i in range(len(block2)):
                self.transition[f"conv{i}"] = deepcopy(transition1.conv)
                norm = deepcopy(block2[f"denselayer{i + 1}"]).norm1
                bn_params = CanonizedDenseBlock.get_bn_params(norm, 0,
                                                              transition1.conv.out_channels)
                assert transition1.conv.out_channels == block2.denselayer1.norm1.num_features

                CanonizedDenseBlock.merge(self.transition[f"conv{i}"], bn_params)
            self.transition[f"conv{len(block2)}"] = deepcopy(transition1.conv)
            if transition2 is not None: # if there is another transition layer after the densblock which sits after this denseblock
                norm = deepcopy(transition2.norm)
            else:
                norm=deepcopy(final_norm)
            bn_params = CanonizedDenseBlock.get_bn_params(norm, 0, transition1.conv.out_channels)
            CanonizedDenseBlock.merge(self.transition[f"conv{len(block2)}"], bn_params)
        else:
            self.transition = None
        self.add_untangled_modules()

    def add_untangled_modules(self):
        if self.initial:
            for i, mod in enumerate(self.initial_bns):
                self.add_module(f'init_bn_{i}', mod)
            for i, mod in enumerate(self.initial_relus):
                self.add_module(f'init_relu_{i}', mod)
        if self.transition is not None:
            for key, elem in self.transition.items():
                self.add_module(f'transition_{key}', elem)
        for i, layer in enumerate(self.layers):
            for key, elem in layer.items():
                if isinstance(elem, torch.nn.Module):
                    self.add_module(f'layer{i}_{key}', elem)

    def initial_forward(self, x):
        out = []
        for i, bn in enumerate(self.initial_bns):
            relu=self.initial_relus[i]
            out.append(relu(bn(x)))
        #out.append(x)
        return out

    def forward(self, x):
        if self.initial:
            x = self.initial_forward(x)

        assert isinstance(x, list)
        assert len(x) == len(self.layers) + 1

        for i, l in enumerate(self.layers):
            temp = ReLU(l["conv1"](x[i]))
            for j in range(l['conv2_count']):
                temp2 = l[f"conv2_{i+j+1}"](temp)
                temp2=ReLU(temp2)
                x[i+j+1] = torch.cat([x[i+j+1], temp2], 1)

        if self.transition is not None:
            return self.transition_forward(x[-1])
        else:
            return x[-1]

    def transition_forward(self, x):
        # forward function for transition layers, outputs a single tensor
        assert self.transition is not None
        assert x.shape[1] == self.transition["conv0"].in_channels
        pool = AvgPool2d(kernel_size=2, stride=2)
        ret = []
        for i in range(len(self.transition.keys())):
            temp=self.transition[f"conv{i}"](x)
            temp=ReLU(pool(temp))
            ret.append(temp)
        return ret

class UntangledDenseNet(Module):
    # New untangled Module that computes the same functions as model. The given DenseNet object should first be canonized by SequentialMergeBatchNorm for this class to compute the correct function
    def __init__(self, model):
        super().__init__()
        self.classifier=deepcopy(model.classifier)
        self.features = Sequential()
        i=0
        block_id=0
        while i < len(model.features):
            mod = model.features[i]
            if isinstance(mod, _DenseBlock):
                block1 = mod
                transition1 = None
                block2 = None
                transition2 = None
                final_norm = None
                # if i < len(model.features) - 1:
                if isinstance(model.features[i + 1], _Transition):
                    transition1 = model.features[i + 1]
                    block2 = model.features[i + 2]
                    if isinstance(model.features[i + 3], _Transition):
                        transition2 = model.features[i + 3]
                    else:
                        assert isinstance(model.features[i + 3], BatchNorm2d)
                        final_norm = model.features[i + 3]
                    i = i + 1
                else:
                    assert isinstance(model.features[i + 1], BatchNorm2d)
                    final_norm = model.features[i + 1]
                    i = i + 1

                self.features.add_module(f"denseblock{block_id}",
                                   CanonizedDenseBlock(block1=block1, transition1=transition1, block2=block2,
                                                       transition2=transition2, initial=(block_id == 0),
                                                       final_norm=final_norm))
                block_id = block_id + 1
            else:
                self.features.append(mod)
            i = i + 1
    def forward(self,x):
        x=self.features(x)
        x=ReLU(x) # unnecessary, relu is already the last operation applied to all channels of the tensor
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x=torch.flatten(x,1)
        return self.classifier(x)


class UntangledDenseNetCanonizer(AttributeCanonizer):
    # Create untangled densenet and call its forward function for forward computation
    def __init__(self):
        super().__init__(self._attribute_map)
        self.canonized_model=None

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, DenseNet):
            attributes = {
                'forward': cls.forward.__get__(module),
                'named_modules':cls.named_modules.__get__(module),
                'canonized_model': UntangledDenseNet(module)
            }
            return attributes
        return None

    def remove(self):
        self.canonized_model=None
        super(UntangledDenseNetCanonizer,self).remove()

    def copy(self):
        '''Copy this Canonizer.

        Returns
        -------
        obj:`Canonizer`
            A copy of this Canonizer.
        '''
        return UntangledDenseNetCanonizer()

    def forward(self, x):
        return self.canonized_model(x)

    def named_modules(self):
        for x in self.canonized_model.named_modules():
            yield x


class InitialBatchNormCanonizer(ThreshReLUMergeBatchNorm):
    #canonizer for initial batchnorm layers of a densenet which has already been modified by an UntangledDenseNetCanonizer
    def apply(self, root_module):
        instances = []
        block=root_module.canonized_model.features.denseblock0
        for i,bn in enumerate(block.initial_bns):
            relu=block.initial_relus[i]
            channels = range(bn.num_features)
            linears=[]
            if i<len(block.layers):
                linears.append(block.layers[i]["conv1"])
            else:
                for key in block.transition.keys():
                    linears.append(block.transition[key])
            instance = self.copy()
            instance.register(tuple(linears), bn, relu, channels)
            instances.append(instance)
        return instances

    def register(self, linears, batch_norm, relu, channels):
        self.relu = relu

        denominator = (batch_norm.running_var + batch_norm.eps) ** .5
        scale = (batch_norm.weight / denominator)  # Weight of the batch norm layer when seen as a linear layer
        shift = batch_norm.bias - batch_norm.running_mean * scale  # bias of the batch norm layer when seen as a linear layer
        self.relu.canonization_params = {}
        self.relu.canonization_params['weights'] = scale
        self.relu.canonization_params['biases'] = shift

        self.linears = linears
        self.batch_norm = batch_norm
        self.linear_params = [(linear.weight.data, getattr(linear.bias, 'data', None)) for linear in linears]

        self.batch_norm_params = {
            key: getattr(self.batch_norm, key).data for key in ('weight', 'bias', 'running_mean', 'running_var')
        }
        returned_handles = self.merge_batch_norm(self.linears, self.batch_norm)
        self.handles = returned_handles

        self.handles.append(self.relu.register_forward_pre_hook(ThreshReLUMergeBatchNorm.prehook))
        self.handles.append(self.relu.register_forward_hook(ThreshReLUMergeBatchNorm.fwdhook))

    def merge_batch_norm(self, modules, batch_norm, channels=None):
        self.batch_norm_eps = batch_norm.eps
        return_handles = []
        denominator = (batch_norm.running_var + batch_norm.eps) ** .5
        scale = (batch_norm.weight / denominator)  # Weight of the batch norm layer when seen as a linear layer
        shift = batch_norm.bias - batch_norm.running_mean * scale  # bias of the batch norm layer when seen as a linear layer

        if channels is None:
            channels=range(batch_norm.num_features)
        for module in modules:
            if isinstance(module, torch.nn.Linear):
                original_weight = module.weight.data
            else:
                original_weight = module.weight.data[:,channels,:,:]
            if module.bias is None:
                module.bias = torch.nn.Parameter(
                    torch.zeros(module.out_channels, device=original_weight.device, dtype=original_weight.dtype)
                )
            original_bias = module.bias.data

            if isinstance(module, ConvolutionTranspose):
                index = (slice(None), *((None,) * (original_weight.ndim - 1)))
            else:
                index = (None, slice(None), *((None,) * (original_weight.ndim - 2)))

            # merge batch_norm into linear layer to the right
            if isinstance(module,torch.nn.Linear):
                module.weight.data=(original_weight*scale[index])
            else:
                module.weight.data[:,channels,:,:] = (original_weight * scale[index])

            # module.bias.data = original_bias
            if isinstance(module, torch.nn.Conv2d):
                if module.padding == (0, 0):
                    module.bias.data = (original_weight * shift[index]).sum(dim=[1, 2, 3]) + original_bias
                else:
                    bias_kernel = shift[index].expand(*(shift[index].shape[0:-2] + original_weight.shape[-2:]))
                    temp_module = torch.nn.Conv2d(in_channels=module.in_channels, out_channels=module.out_channels,
                                                  kernel_size=module.kernel_size, padding=module.padding,padding_mode=module.padding_mode, bias=False)
                    temp_module.weight.data[:,channels,:,:] = original_weight
                    bias_kernel = temp_module(bias_kernel).detach()

                    module.canonization_params = {}
                    module.canonization_params["bias_kernel"] = bias_kernel
                    return_handles.append(module.register_forward_hook(SequentialMergeBatchNormtoRight.convhook))
            elif isinstance(module, torch.nn.Linear):
                module.bias.data = (original_weight * shift).sum(dim=1) + original_bias

        # change batch_norm parameters to produce identity
        batch_norm.running_mean.data = torch.zeros_like(batch_norm.running_mean.data)
        batch_norm.running_var.data = torch.ones_like(batch_norm.running_var.data)
        batch_norm.bias.data = torch.zeros_like(batch_norm.bias.data)
        batch_norm.weight.data = torch.ones_like(batch_norm.weight.data)
        batch_norm.eps = 0.
        return return_handles

class UntanglingCanonizer(CorrectCompositeCanonizer):
    # overall wrapper for untangling densenets
    def __init__(self, canonize_initial=True):
        if canonize_initial:
            super().__init__((
                CorrectSequentialMergeBatchNorm(),
                UntangledDenseNetCanonizer(),
                InitialBatchNormCanonizer()
            ))
        else:
            super().__init__((
                CorrectSequentialMergeBatchNorm(),
                UntangledDenseNetCanonizer(),
            ))

    def copy(self):
        return UntanglingCanonizer()



if __name__=='__main__':

    N = 100
    torch.random.manual_seed(42)
    cont=True

    model = torchvision.models.densenet.DenseNet(2, (5, 6, 9, 6), 5)
    for mod in model.features:
        if isinstance(mod, _DenseBlock):
            for u in range(len(mod)):
                bn = mod[f"denselayer{u + 1}"].norm1
                bn.weight.data = torch.normal(mean=0, std=1,size=bn.weight.data.shape)
                bn.bias.data = torch.normal(mean=0, std=1,size=bn.bias.data.shape)
                bn = mod[f"denselayer{u + 1}"].norm2
                bn.weight.data = torch.normal(mean=0, std=1,size=bn.weight.data.shape)
                bn.bias.data = torch.normal(mean=0, std=1,size=bn.bias.data.shape)
                bn.running_var=torch.rand(bn.running_var.data.shape[0])
                bn.running_mean=torch.rand(bn.running_mean.data.shape[0])
        elif isinstance(mod,_Transition):
            bn=mod.norm
            bn.weight.data = torch.normal(mean=0, std=1, size=bn.weight.data.shape)
            bn.bias.data = torch.normal(mean=0, std=1, size=bn.bias.data.shape)
            bn.running_mean=torch.rand(bn.running_mean.data.shape[0])
            bn.running_var=torch.rand(bn.running_var.data.shape[0])
        elif isinstance(mod, BatchNorm2d):
            bn=mod
            bn.weight.data = torch.normal(mean=0, std=1,size=bn.weight.data.shape)
            bn.bias.data = torch.normal(mean=0, std=1,size=bn.bias.data.shape)
            bn.running_mean=torch.rand(bn.running_mean.data.shape[0])
            bn.running_var=torch.rand(bn.running_var.data.shape[0])
    model = torchvision.models.densenet121(pretrained=True)
    for leaf in collect_leaves(model):
        print(leaf)
    model.to(torch.device('cpu'))
    model.eval()
    while(cont):
        x = torch.rand(N, 3, 224, 224, device="cpu")
        with torch.no_grad():
            y1 = model(x)

        #canon = UntanglingCanonizer(canonize_initial=False)
        canon = UntanglingCanonizer(canonize_initial=True)
        #canon = CorrectSequentialMergeBatchNorm()

        handles = canon.apply(model)
        with torch.no_grad():
            y2 = model(x)

        print(f"{torch.norm(y2 - y1)} for input with norm {torch.norm(x)}")

        for h in handles:
            h.remove()
        with torch.no_grad():
            y2 = model(x)
        print(f"{torch.norm(y2 - y1)} after detach")
        print("\n\n=====\n\n")
        a=input()
        cont=(a!="q")

