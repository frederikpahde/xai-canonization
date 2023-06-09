import torch
from zennit.core import collect_leaves
from zennit.types import Convolution, BatchNorm
from zennit import canonizers as zcanonizers

class MergeBNConv(zcanonizers.Canonizer):
    '''Abstract Canonizer to merge the parameters of batch norms into linear modules.'''
    linear_type = (
        Convolution,
    )
    batch_norm_type = (
        BatchNorm,
    )

    def __init__(self):
        super().__init__()
        self.linears = None
        self.batch_norm = None

        self.linear_params = None
        self.batch_norm_params = None

    def register(self, linears, batch_norm):
        '''Store the parameters of the linear modules and the batch norm module and apply the merge.

        Parameters
        ----------
        linear: list of obj:`torch.nn.Module`
            List of linear layer with mandatory attributes `weight` and `bias`.
        batch_norm: obj:`torch.nn.Module`
            Batch Normalization module with mandatory attributes
            `running_mean`, `running_var`, `weight`, `bias` and `eps`
        '''
        self.linears = linears
        self.batch_norm = batch_norm

        self.linear_params = [(linear.weight.data, getattr(linear.bias, 'data', None)) for linear in linears]

        self.batch_norm_params = {
            key: getattr(self.batch_norm, key).data for key in ('weight', 'bias', 'running_mean', 'running_var')
        }

        self.merge_batch_norm(self.linears, self.batch_norm)

    def remove(self):
        '''Undo the merge by reverting the parameters of both the linear and the batch norm modules to the state before
        the merge.
        '''
        for linear, (weight, bias) in zip(self.linears, self.linear_params):
            linear.weight.data = weight
            if bias is None:
                linear.bias = None
            else:
                linear.bias.data = bias

        for key, value in self.batch_norm_params.items():
            getattr(self.batch_norm, key).data = value

    @staticmethod
    def merge_batch_norm(modules, batch_norm):
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
        # denominator = (batch_norm.running_var + batch_norm.eps) ** .5
        
        ## BN transformation as linear transformation
        bn_weight = batch_norm.weight.data / ((batch_norm.running_var.data + batch_norm.eps) ** 0.5)
        bn_bias = batch_norm.bias.data - batch_norm.running_mean.data * bn_weight

        for module in modules:
            original_weight = module.weight.data
            if module.bias is None:
                module.bias = torch.nn.Parameter(
                    torch.zeros(1, device=original_weight.device, dtype=original_weight.dtype)
                )
            original_bias = module.bias.data

            ## Computation Weight
            weight_new = original_weight * bn_weight[None, :, None, None]

            ## Computation Bias
            bias_new = (original_weight * bn_bias[None, :, None, None]).sum(axis=(1,2,3)) + original_bias

            # merge batch_norm into linear layer
            module.weight.data = weight_new
            module.bias.data = bias_new

        # change batch_norm parameters to produce identity
        batch_norm.running_mean.data = torch.zeros_like(batch_norm.running_mean.data)
        batch_norm.running_var.data = torch.ones_like(batch_norm.running_var.data)
        batch_norm.bias.data = torch.zeros_like(batch_norm.bias.data)
        batch_norm.weight.data = torch.ones_like(batch_norm.weight.data)

class SequentialMergeBNConv(MergeBNConv):
    '''Canonizer to merge the parameters of all batch norms that appear sequentially right after a linear module.

    Note
    ----
    SequentialMergeBatchNorm traverses the tree of children of the provided module depth-first and in-order.
    This means that child-modules must be assigned to their parent module in the order they are visited in the forward
    pass to correctly identify adjacent modules.
    This also means that activation functions must be assigned in their module-form as a child to their parent-module
    to properly detect when there is an activation function between linear and batch-norm modules.

    '''
    def apply(self, root_module):
        '''Finds a batch norm following right after a linear layer, and creates a copy of this instance to merge
        them by fusing the batch norm parameters into the linear layer and reducing the batch norm to the identity.

        Parameters
        ----------
        root_module: obj:`torch.nn.Module`
            A module of which the leaves will be searched and if a batch norm is found right after a linear layer, will
            be merged.

        Returns
        -------
        instances: list
            A list of instances of this class which modified the appropriate leaves.
        '''
        instances = []
        last_leaf = None
        for leaf in collect_leaves(root_module):
            if isinstance(leaf, self.linear_type):
                if isinstance(last_leaf, self.batch_norm_type):
                    instance = self.copy()
                    instance.register((leaf,), last_leaf)
                    instances.append(instance)
                else:
                    pass
            last_leaf = leaf

        return instances