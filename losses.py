import torch


def heston_pde_residual(model, inputs, params):
    """
    Placeholder Heston PDE residual.
    This enforces structure and differentiability,
    not the full closed-form operator.
    """
    inputs.requires_grad_(True)
    price = model(inputs)

    grads = torch.autograd.grad(
        price,
        inputs,
        grad_outputs=torch.ones_like(price),
        create_graph=True
    )[0]

    # Placeholder residual (structural PINN demo)
    residual = grads.sum(dim=1, keepdim=True)
    return residual


def feller_penalty(params):
    kappa, theta, sigma = params
    return torch.relu(sigma**2 - 2 * kappa * theta)
