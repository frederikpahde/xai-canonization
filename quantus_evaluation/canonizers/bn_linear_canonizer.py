import copy
import torch
from zennit import canonizers as zcanonizers


class MergeBatchNormByIndices(zcanonizers.Canonizer):

    def __init__(self):
        super().__init__()
        self.linear_params = None
        self.batch_norm_params = None

    def register(self, linear, batch_norm, indices):

        self.linear = linear
        self.batch_norm = batch_norm
        self.indices = indices

        self.linear_params = (linear.weight.data, getattr(linear.bias, 'data', None))

        self.batch_norm_params = {
            key: getattr(self.batch_norm, key).data for key in ('weight', 'bias', 'running_mean', 'running_var')
        }

        self.merge_batch_norm_by_indices(self.linear, self.batch_norm, self.indices)

    def remove(self):
        '''Undo the merge by reverting the parameters of both the linear and the batch norm modules to the state before
        the merge.
        '''
        weight, bias = self.linear_params
        self.linear.weight.data = weight
        if bias is None:
            self.linear.bias = None
        else:
            self.linear.bias.data = bias

        for key, value in self.batch_norm_params.items():
            getattr(self.batch_norm, key).data = value

    @staticmethod
    def merge_batch_norm_by_indices(linear, batch_norm, indices):
        '''Update parameters of a linear layer to additionally include a Batch Normalization operation and update the
        batch normalization layer to instead compute the identity.

        Parameters
        ----------
        modules: list of obj:`torch.nn.Module`
            Linear layers with mandatory attributes `weight` and `bias`.
        batch_norm: obj:`torch.nn.Module`
            Batch Normalization module with mandatory attributes `running_mean`, `running_var`, `weight`, `bias` and
            `eps`
        '''
        w_linbn = batch_norm.weight.data / ((batch_norm.running_var.data + batch_norm.eps) ** 0.5)
        b_linbn = batch_norm.bias.data - (batch_norm.running_mean.data * w_linbn)

        original_weight = copy.deepcopy(linear.weight.data)
        original_bias = copy.deepcopy(linear.bias.data)

        w_new = copy.deepcopy(original_weight)
        b_new = copy.deepcopy(original_bias)

        for inds in indices:
            original_weight_rel = copy.deepcopy(original_weight)[:, inds[0]:inds[1]]
            w_new_rel = copy.deepcopy(original_weight_rel) * w_linbn[None, :]
            w_new[:, inds[0]:inds[1]] = w_new_rel

            b_new_rel = copy.deepcopy(original_weight_rel) * b_linbn[None, :]
            b_new += b_new_rel.sum(axis=1)
            
        linear.weight.data = w_new
        linear.bias.data = b_new

        # change batch_norm parameters to produce identity
        batch_norm.running_mean.data = torch.zeros_like(batch_norm.running_mean.data)
        batch_norm.running_var.data = torch.ones_like(batch_norm.running_var.data)
        batch_norm.bias.data = torch.zeros_like(batch_norm.bias.data)
        batch_norm.weight.data = torch.ones_like(batch_norm.weight.data)


class NamedMergeBatchNormByIndices(MergeBatchNormByIndices):
    def __init__(self, name_map_with_indices):
        super().__init__()
        self.name_map_with_indices = name_map_with_indices

    def apply(self, root_module):
        instances = []
        lookup = dict(root_module.named_modules())
        for batch_norm_name, (linear_name, indices),  in self.name_map_with_indices.items():
            instance = self.copy()
            instance.register(lookup[linear_name], lookup[batch_norm_name], indices)
            instances.append(instance)

        return instances

    def copy(self):
        return self.__class__(self.name_map_with_indices)