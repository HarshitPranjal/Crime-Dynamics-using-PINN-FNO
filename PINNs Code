# ==========================================================
# INVERSE PINN FOR NCRB CRIME DATA
# ESTIMATING:
# alpha, l, delta0, delta1, A
# ==========================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)

# ==========================================================
# NCRB DATA
# ==========================================================

crime_data = np.array([

1740000,1782000,1834000,1893000,1956000,
2011000,2074000,2129000,2178000,
2221000,2265000,2308000,2352000,
2394000,2439000,2483000,2526000,
2571000,2617000,2663000,
2708000,2754000,2801000,2849000,
2898000,2946000,2995000,
3043000,3092000,3141000,
3189000,3237000,3284000,
3332000,3379000,3426000,
3473000,3519000,2935000

], dtype=np.float32)

# ==========================================================
# NORMALIZATION
# ==========================================================

crime_data = (
    crime_data - crime_data.min()
) / (
    crime_data.max() - crime_data.min()
)

years = np.arange(len(crime_data))

t_data = torch.tensor(
    years / years.max(),
    dtype=torch.float32
).view(-1,1).to(device)

c_data = torch.tensor(
    crime_data,
    dtype=torch.float32
).view(-1,1).to(device)

# ==========================================================
# PINN
# ==========================================================

class CrimePINN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(1,64),
            nn.Tanh(),

            nn.Linear(64,64),
            nn.Tanh(),

            nn.Linear(64,64),
            nn.Tanh(),

            nn.Linear(64,64),
            nn.Tanh(),

            nn.Linear(64,3)
        )

        # Raw trainable parameters

        self.alpha_raw  = nn.Parameter(torch.tensor(0.0))
        self.l_raw      = nn.Parameter(torch.tensor(0.0))
        self.delta0_raw = nn.Parameter(torch.tensor(0.0))
        self.delta1_raw = nn.Parameter(torch.tensor(0.0))
        self.A_raw      = nn.Parameter(torch.tensor(0.0))

        self.initialize()

    def initialize(self):

        for m in self.modules():

            if isinstance(m, nn.Linear):

                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,t):

        out = self.net(t)

        # Force S,C,P in (0,1)

        return torch.sigmoid(out)

    # ----------------------------------------
    # Parameters constrained to (0,1)
    # ----------------------------------------

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)

    @property
    def l(self):
        return torch.sigmoid(self.l_raw)

    @property
    def delta0(self):
        return torch.sigmoid(self.delta0_raw)

    @property
    def delta1(self):
        return torch.sigmoid(self.delta1_raw)

    @property
    def A(self):
        return torch.sigmoid(self.A_raw)

# ==========================================================
# MODEL
# ==========================================================

model = CrimePINN().to(device)

# ==========================================================
# PDE RESIDUALS
# ==========================================================

def pde_residual(t):

    t.requires_grad_(True)

    pred = model(t)

    S = pred[:,0:1]
    C = pred[:,1:2]
    P = pred[:,2:3]

    S_t = torch.autograd.grad(
        S,
        t,
        grad_outputs=torch.ones_like(S),
        create_graph=True
    )[0]

    C_t = torch.autograd.grad(
        C,
        t,
        grad_outputs=torch.ones_like(C),
        create_graph=True
    )[0]

    P_t = torch.autograd.grad(
        P,
        t,
        grad_outputs=torch.ones_like(P),
        create_graph=True
    )[0]

    alpha  = model.alpha
    l       = model.l
    delta0  = model.delta0
    delta1  = model.delta1
    A       = model.A

    # ODE system obtained after spatial averaging

    R1 = (
        S_t
        - A
        + alpha*S*C
        - delta1*P
        + delta0*S
    )

    R2 = (
        C_t
        - alpha*S*C
        + l*C
        + delta0*C
    )

    R3 = (
        P_t
        - l*C
        + delta1*P
        + delta0*P
    )

    return R1,R2,R3

# ==========================================================
# COLLOCATION POINTS
# ==========================================================

Nf = 5000

t_f = torch.rand(
    (Nf,1),
    device=device
)

# ==========================================================
# INITIAL CONDITIONS
# ==========================================================

t0 = torch.zeros(
    (1,1),
    device=device
)

S0 = torch.tensor(
    [[0.95]],
    dtype=torch.float32,
    device=device
)

C0 = torch.tensor(
    [[crime_data[0]]],
    dtype=torch.float32,
    device=device
)

P0 = torch.tensor(
    [[0.0]],
    dtype=torch.float32,
    device=device
)

# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

# ==========================================================
# TRAINING
# ==========================================================

epochs = 15000

loss_history = []

print("Training Inverse PINN...\n")

for epoch in range(epochs):

    optimizer.zero_grad()

    # --------------------------------------
    # PDE LOSS
    # --------------------------------------

    R1,R2,R3 = pde_residual(t_f)

    loss_pde = (

        torch.mean(R1**2)
        +
        torch.mean(R2**2)
        +
        torch.mean(R3**2)

    )

    # --------------------------------------
    # DATA LOSS
    # --------------------------------------

    pred_data = model(t_data)

    C_pred = pred_data[:,1:2]

    loss_data = torch.mean(
        (C_pred - c_data)**2
    )

    # --------------------------------------
    # INITIAL CONDITION LOSS
    # --------------------------------------

    pred0 = model(t0)

    loss_ic = (

        torch.mean(
            (pred0[:,0:1]-S0)**2
        )

        +

        torch.mean(
            (pred0[:,1:2]-C0)**2
        )

        +

        torch.mean(
            (pred0[:,2:3]-P0)**2
        )

    )

    # --------------------------------------
    # TOTAL LOSS
    # --------------------------------------

    loss = (

        loss_pde

        + 100*loss_data

        + 100*loss_ic

    )

    loss.backward()

    optimizer.step()

    loss_history.append(
        loss.item()
    )

    if epoch % 500 == 0:

        print(
            f"Epoch {epoch:5d}"
            f" | Loss = {loss.item():.6e}"
        )

# ==========================================================
# ESTIMATED PARAMETERS
# ==========================================================

print("\nEstimated Parameters\n")

print("alpha  =", model.alpha.item())
print("l      =", model.l.item())
print("delta0 =", model.delta0.item())
print("delta1 =", model.delta1.item())
print("A      =", model.A.item())

# ==========================================================
# PREDICTIONS
# ==========================================================

with torch.no_grad():

    pred = model(t_data)

S_pred = pred[:,0].cpu().numpy()
C_pred = pred[:,1].cpu().numpy()
P_pred = pred[:,2].cpu().numpy()

# ==========================================================
# LOSS CURVE
# ==========================================================

plt.figure(figsize=(7,5))

plt.semilogy(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Inverse PINN Training Loss")

plt.grid(True)

plt.show()

# ==========================================================
# NCRB FIT
# ==========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    years,
    crime_data,
    s=40,
    label="NCRB Data"
)

plt.plot(
    years,
    C_pred,
    linewidth=3,
    label="PINN Fit"
)

plt.xlabel("Year Index")
plt.ylabel("Normalized Crime")

plt.title("Inverse PINN Fit to NCRB Crime Data")

plt.legend()
plt.grid(True)

plt.show()
