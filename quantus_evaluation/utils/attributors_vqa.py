import torch
from zennit.attribution import Attributor, constant

class GradientVQA(Attributor):

    def __init__(self, model, composite=None, attr_output=None, create_graph=False, retain_graph=None):
        super().__init__(model=model, composite=composite, attr_output=attr_output)
        self.create_graph = create_graph
        self.retain_graph = retain_graph

    def forward(self, input, attr_output_fn):

        input_img, input_q, input_lengths = input
        input_img = input_img.detach().requires_grad_(True)
        output = self.model(input_img, input_q, input_lengths)
        gradient, = torch.autograd.grad((output,), (input_img,), grad_outputs=(attr_output_fn(output.detach()),))
        return output, gradient

    def grad(self, input, attr_output_fn):
        input_img, input_q, input_lengths = input
        if not input_img.requires_grad:
            input_img.requires_grad = True
        output = self.model(input_img, input_q, input_lengths)
        gradient, = torch.autograd.grad(
            (output,),
            (input_img,),
            grad_outputs=(attr_output_fn(output),),
            create_graph=self.create_graph,
            retain_graph=self.retain_graph,
        )
        return output, gradient

class IntegratedGradientsVQA(GradientVQA):

    def __init__(
        self,
        model,
        composite=None,
        attr_output=None,
        create_graph=False,
        retain_graph=None,
        baseline_fn=None,
        n_iter=20
    ):
        super().__init__(
            model=model,
            composite=composite,
            attr_output=attr_output,
            create_graph=create_graph,
            retain_graph=retain_graph
        )
        if baseline_fn is None:
            baseline_fn = torch.zeros_like
        self.baseline_fn = baseline_fn
        self.n_iter = n_iter

    def forward(self, input, attr_output_fn):
        input_img, input_q, input_lengths = input
        
        baseline = self.baseline_fn(input_img)
        result = torch.zeros_like(input_img)
        
        for alpha in torch.linspace(1. / self.n_iter, 1., self.n_iter):
            path_step = baseline + alpha * (input_img - baseline)
            output, gradient = self.grad((path_step, input_q, input_lengths), attr_output_fn)
            result += gradient / self.n_iter

        result *= (input_img - baseline)
        # in the last step, path_step is equal to input, thus `output` is the original output
        return output, result

class SmoothGradVQA(GradientVQA):
    def __init__(
        self, model, composite=None, attr_output=None, create_graph=False, retain_graph=None, noise_level=0.05, n_iter=300
    ):
        super().__init__(
            model=model,
            composite=composite,
            attr_output=attr_output,
            create_graph=create_graph,
            retain_graph=retain_graph
        )
        self.noise_level = noise_level
        self.n_iter = n_iter

    def forward(self, input, attr_output_fn):
        input_img, input_q, input_lengths = input
        
        dims = tuple(range(1, input_img.ndim))
        std = self.noise_level * (input_img.amax(dims, keepdim=True) - input_img.amin(dims, keepdim=True))

        result = torch.zeros_like(input_img)
        for n in range(self.n_iter):
            # and have SmoothGrad w/ n_iter = 1 === gradient
            if n == self.n_iter - 1:
                epsilon = torch.zeros_like(input_img)
            else:
                epsilon = torch.randn_like(input_img) * std
                
            output, gradient = self.grad((input_img + epsilon, input_q, input_lengths), attr_output_fn)
            result += gradient / self.n_iter

        # output is leaking from the loop for the last epsilon (which is zero)
        return output, result
