import torch
import matplotlib.pyplot as plt

from model import PINN
from losses import heston_pde_residual, feller_penalty
from data import sample_domain


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Model
# -------------------------
model = PINN(input_dim=3).to(device)

# Trainable Heston parameters
params = [
    torch.tensor(2.0, requires_grad=True, device=device),   # kappa
    torch.tensor(0.04, requires_grad=True, device=device),  # theta
    torch.tensor(0.3, requires_grad=True, device=device)    # sigma
]

optimizer = torch.optim.Adam(
    list(model.parameters()) + params, lr=1e-3
)

# -------------------------
# History containers
# -------------------------
loss_history = []
pde_history = []
feller_history = []

# -------------------------
# Training loop
# -------------------------
n_steps = 3000

for step in range(n_steps):
    optimizer.zero_grad()

    x = sample_domain(1024).to(device)
    residual = heston_pde_residual(model, x, params)

    loss_pde = torch.mean(residual**2)
    loss_feller = feller_penalty(params)

    loss = loss_pde + 10.0 * loss_feller
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    pde_history.append(loss_pde.item())
    feller_history.append(loss_feller.item())

    if step % 500 == 0:
        print(
            f"step {step:4d} | "
            f"total {loss.item():.3e} | "
            f"pde {loss_pde.item():.3e} | "
            f"feller {loss_feller.item():.3e}"
        )

# -------------------------
# Plot diagnostics
# -------------------------
plt.figure()
plt.semilogy(loss_history, label="total loss")
plt.semilogy(pde_history, label="PDE loss")
plt.semilogy(feller_history, label="Feller penalty")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Heston PINN training diagnostics")
plt.legend()
plt.tight_layout()

plt.savefig("results/training_diagnostics.png", dpi=150)
plt.show()
