import torch

def count_model_parameters(model: torch.nn.Module) -> int:

    r"""A function to count model parameters"""

    n_parameters = 0

    for parameter in model.parameters():

        if parameter.requires_grad:

            n_p = 1
            for m in range(len(parameter.shape)):

                n_p *= parameter.shape[m]

            n_parameters += n_p

    return n_parameters
