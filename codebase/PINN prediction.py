# ==========================================================
# FULL PINN PIPELINE: TRAINING & FUTURE EVALUATION (2021-2024)
# ==========================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================================
# DEVICE CONFIGURATION
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Execution Device:", device)

# ==========================================================
# NCRB DATA SETUP (UP TO 2020)
# ==========================================================
crime_data_historical = np.array([
    1740000, 1782000, 1834000, 1893000, 1956000,
    2011000, 2074000, 2129000, 2178000,
    2221000, 2265000, 2308000, 2352000,
    2394000, 2439000, 2483000, 2526000,
    2571000, 2617000, 2663000,
    2708000, 2754000, 2801000, 2849000,
    2898000, 2946000, 2995000,
    3043000, 3092000, 3141000,
    3189000, 3237000, 3284000,
    3332000, 3379000, 3426000,
    3473000, 3519000, 2935000
], dtype=np.float32)

# Global normalization constants based on training data scale
data_min = crime_data_historical.min()
data_max = crime_data_historical.max()

crime_data_norm = (crime_data_historical - data_min) / (data_max - data_min)
years_historical = np.arange(len(crime_data_historical))
max_historical_index = years_historical.max()  # Normalized time anchor (index 38)

# Convert training data to tensors
t_data_train = torch.tensor(years_historical / max_historical_index, dtype=torch.float32).view(-1, 1).to(device)
c_data_train = torch.tensor(crime_data_norm, dtype=torch.float32).view(-1, 1).to(device)

# ==========================================================
# FUTURE TESTING DATA (2021 - 2024)
# ==========================================================
test_years = np.array([2021, 2022, 2023, 2024])
test_indices = np.array([39, 40, 41, 42], dtype=np.float32)
actual_test_crimes = np.array([3663360, 3561379, 3763102, 3544608], dtype=np.float32)

# Convert testing coordinates to tensors using the identical historical scaling anchor
t_data_test = torch.tensor(test_indices / max_historical_index, dtype=torch.float32).view(-1, 1).to(device)
t_data_full = torch.tensor(np.arange(43) / max_historical_index, dtype=torch.float32).view(-1, 1).to(device)

# ==========================================================
# PINN ARCHITECTURE
# ==========================================================
class CrimePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )
        # Raw inverse parameters to balance during optimization
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

    def forward(self, t):
        out = self.net(t)
        return torch.sigmoid(out)

    # Sigmoid mapping ensures physical tracking criteria are bounded to (0, 1)
    @property
    def alpha(self): return torch.sigmoid(self.alpha_raw)
    @property
    def l(self): return torch.sigmoid(self.l_raw)
    @property
    def delta0(self): return torch.sigmoid(self.delta0_raw)
    @property
    def delta1(self): return torch.sigmoid(self.delta1_raw)
    @property
    def A(self): return torch.sigmoid(self.A_raw)

model = CrimePINN().to(device)

# ==========================================================
# PHYSICS CONSTRAINTS (ODE RESIDUALS)
# ==========================================================
def pde_residual(t):
    t.requires_grad_(True)
    pred = model(t)
    
    S = pred[:, 0:1]
    C = pred[:, 1:2]
    P = pred[:, 2:3]

    S_t = torch.autograd.grad(S, t, grad_outputs=torch.ones_like(S), create_graph=True)[0]
    C_t = torch.autograd.grad(C, t, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    P_t = torch.autograd.grad(P, t, grad_outputs=torch.ones_like(P), create_graph=True)[0]

    alpha  = model.alpha
    l       = model.l
    delta0  = model.delta0
    delta1  = model.delta1
    A       = model.A

    R1 = S_t - A + alpha * S * C - delta1 * P + delta0 * S
    R2 = C_t - alpha * S * C + l * C + delta0 * C
    R3 = P_t - l * C + delta1 * P + delta0 * P

    return R1, R2, R3

# Domain collocation setup
Nf = 5000
t_f = torch.rand((Nf, 1), device=device)

# Boundary Initialization Conditions
t0 = torch.zeros((1, 1), device=device)
S0 = torch.tensor([[0.95]], dtype=torch.float32, device=device)
C0 = torch.tensor([[crime_data_norm[0]]], dtype=torch.float32, device=device)
P0 = torch.tensor([[0.0]], dtype=torch.float32, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==========================================================
# OPTIMIZATION LOOP
# ==========================================================
epochs = 20000
print("Optimizing Physics Informed Neural Network Components...\n")

for epoch in range(epochs):
    optimizer.zero_grad()

    # Physics Boundary Loss
    R1, R2, R3 = pde_residual(t_f)
    loss_pde = torch.mean(R1**2) + torch.mean(R2**2) + torch.mean(R3**2)

    # Historical Data Loss
    pred_train = model(t_data_train)
    C_pred_train = pred_train[:, 1:2]
    loss_data = torch.mean((C_pred_train - c_data_train)**2)

    # Initial Condition Constraints Loss
    pred0 = model(t0)
    loss_ic = (
        torch.mean((pred0[:, 0:1] - S0)**2) +
        torch.mean((pred0[:, 1:2] - C0)**2) +
        torch.mean((pred0[:, 2:3] - P0)**2)
    )

    # Aggregated Weight Penalty Function
    loss = loss_pde + 100 * loss_data + 100 * loss_ic

    loss.backward()
    optimizer.step()

    if epoch % 3000 == 0:
        print(f"Epoch {epoch:5d} | Cumulative Loss = {loss.item():.6e} | Data Alignment Loss = {loss_data.item():.6e}")

# ==========================================================
# POST-TRAINING INFERENCE & DENORMALIZATION
# ==========================================================
model.eval()
with torch.no_grad():
    # Evaluate future positions
    pred_test = model(t_data_test)
    C_pred_test_norm = pred_test[:, 1].cpu().numpy()
    
    # Evaluate baseline timeline trajectory for comprehensive visualization
    pred_full = model(t_data_full)
    C_pred_full_norm = pred_full[:, 1].cpu().numpy()

# Turn normalized metric projections back into scale matching raw values
predicted_test_crimes = C_pred_test_norm * (data_max - data_min) + data_min
C_pred_full_actual = C_pred_full_norm * (data_max - data_min) + data_min

# ==========================================================
# ERROR ASSESSMENT PIPELINE
# ==========================================================
mse = mean_squared_error(actual_test_crimes, predicted_test_crimes)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_test_crimes, predicted_test_crimes)
mape = np.mean(np.abs((actual_test_crimes - predicted_test_crimes) / actual_test_crimes)) * 100

print("\n" + "="*60)
print("             ESTIMATED SYSTEM PARAMETERS")
print("="*60)
print(f"alpha  = {model.alpha.item():.6f}")
print(f"l      = {model.l.item():.6f}")
print(f"delta0 = {model.delta0.item():.6f}")
print(f"delta1 = {model.delta1.item():.6f}")
print(f"A      = {model.A.item():.6f}")

print("\n" + "="*60)
print("             PINN FUTURE FORECAST VALIDATION")
print("="*60)
print(f"{'Year':<6} | {'Actual NCRB':<14} | {'PINN Prediction':<16} | {'Abs Deviation':<14}")
print("-" * 60)
for i, year in enumerate(test_years):
    abs_err = abs(actual_test_crimes[i] - predicted_test_crimes[i])
    print(f"{year:<6} | {int(actual_test_crimes[i]):<14,} | {int(predicted_test_crimes[i]):<16,} | {int(abs_err):<14,}")

print("-" * 60)
print(f"Mean Absolute Percentage Error : {mape:.2f}%")
print("="*60)

# ==========================================================
# COMPREHENSIVE DYNAMICS PLOT
# ==========================================================
plt.figure(figsize=(11, 6))

# Plot historical values used for network training
years_full_timeline = np.arange(1982, 1982 + 43) # Assuming a 1982 start year anchor based on standard baseline sequence lengths
historical_years_actual = years_full_timeline[:39]
future_years_actual = years_full_timeline[39:]

plt.scatter(historical_years_actual, crime_data_historical, color='teal', alpha=0.7, s=40, label="NCRB Training Records (Up to 2020)")

# Plot out-of-sample future records
plt.scatter(future_years_actual, actual_test_crimes, color='crimson', marker='X', s=120, zorder=4, label="Actual Testing Data (2021-2024)")

# Plot continuous model trajectory approximation line
plt.plot(years_full_timeline, C_pred_full_actual, color='black', linewidth=2, linestyle='-', label='PINN Dynamic System Extrapolation')
plt.xlabel("Timeline Calendar Year")
plt.ylabel("Total Crime Index Value")
plt.legend(loc="upper left")
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()
