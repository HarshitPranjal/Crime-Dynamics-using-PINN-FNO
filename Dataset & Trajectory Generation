import numpy as np
import os
from tqdm import tqdm

# =====================================================
# SAVE DIRECTORY
# =====================================================

SAVE_DIR = "crime_fno_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# ESTIMATED PARAMETERS
# =====================================================

ALPHA_EST  = 0.99907994
L_EST      = 0.00106983
DELTA0_EST = 0.00109562
DELTA1_EST = 0.54151040
A_EST      = 0.34311762

LAMBDA = 0.01

# =====================================================
# GRID
# =====================================================

Nx = 32
Ny = 32

Lx = 1.0
Ly = 1.0

dx = Lx/(Nx-1)
dy = Ly/(Ny-1)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

X, Y = np.meshgrid(x, y)

# =====================================================
# TIME
# =====================================================

dt = 0.001
T = 2.0

Nt = int(T/dt)

# =====================================================
# NUMBER OF SIMULATIONS
# =====================================================

NUM_SIMULATIONS = 5000

# =====================================================
# NEUMANN LAPLACIAN
# =====================================================

def laplacian_neumann(U):

    U_pad = np.pad(
        U,
        ((1,1),(1,1)),
        mode="edge"
    )

    lap = (
        U_pad[2:,1:-1]
        + U_pad[:-2,1:-1]
        + U_pad[1:-1,2:]
        + U_pad[1:-1,:-2]
        - 4*U
    )/(dx*dx)

    return lap

# =====================================================
# STORAGE
# =====================================================

inputs = []
targets = []

# =====================================================
# MAIN LOOP
# =====================================================

print("Generating dataset...")

for sim in tqdm(range(NUM_SIMULATIONS)):

    # -------------------------------------------------
    # PARAMETER PERTURBATION
    # -------------------------------------------------

    alpha  = ALPHA_EST  * np.random.uniform(0.9, 1.1)
    l      = L_EST      * np.random.uniform(0.9, 1.1)
    delta0 = DELTA0_EST * np.random.uniform(0.9, 1.1)
    delta1 = DELTA1_EST * np.random.uniform(0.9, 1.1)
    A      = A_EST      * np.random.uniform(0.9, 1.1)

    # -------------------------------------------------
    # INITIAL CONDITIONS
    # -------------------------------------------------

    S = np.ones((Nx,Ny)) * np.random.uniform(0.8,1.0)

    C = np.zeros((Nx,Ny))
    P = np.zeros((Nx,Ny))

    # -------------------------------------------------
    # RANDOM CRIME HOTSPOT
    # -------------------------------------------------

    x0 = np.random.uniform(0.2,0.8)
    y0 = np.random.uniform(0.2,0.8)

    sigma = np.random.uniform(0.03,0.12)
    amp   = np.random.uniform(0.1,0.4)

    C += amp*np.exp(
        -(
            (X-x0)**2 +
            (Y-y0)**2
        )/(2*sigma**2)
    )

    # -------------------------------------------------
    # INPUT SAMPLE
    # -------------------------------------------------

    input_sample = np.stack([

        S,
        C,
        P,

        np.ones_like(S)*alpha,
        np.ones_like(S)*l,

        np.ones_like(S)*delta0,
        np.ones_like(S)*delta1,

        np.ones_like(S)*A

    ], axis=0)

    # -------------------------------------------------
    # TIME INTEGRATION
    # -------------------------------------------------

    for n in range(Nt):

        lapS = laplacian_neumann(S)
        lapC = laplacian_neumann(C)
        lapP = laplacian_neumann(P)

        dS = (
            LAMBDA*lapS
            + A
            - alpha*S*C
            + delta1*P
            - delta0*S
        )

        dC = (
            LAMBDA*lapC
            + alpha*S*C
            - l*C
            - delta0*C
        )

        dP = (
            LAMBDA*lapP
            + l*C
            - delta1*P
            - delta0*P
        )

        S += dt*dS
        C += dt*dC
        P += dt*dP

        S = np.maximum(S,0)
        C = np.maximum(C,0)
        P = np.maximum(P,0)

    # -------------------------------------------------
    # STORE FINAL STATE ONLY
    # -------------------------------------------------

    future_state = np.stack(
        [S,C,P],
        axis=0
    ).astype(np.float32)

    inputs.append(
        input_sample.astype(np.float32)
    )

    targets.append(
        future_state
    )

# =====================================================
# CONVERT TO ARRAYS
# =====================================================

inputs = np.array(inputs, dtype=np.float32)
targets = np.array(targets, dtype=np.float32)

print("\\nInputs Shape")
print(inputs.shape)

print("\\nTargets Shape")
print(targets.shape)

# Expected:
# Inputs  -> (5000, 8, 32, 32)
# Targets -> (5000, 3, 32, 32)

# =====================================================
# SAVE DATASET
# =====================================================

np.save(
    os.path.join(SAVE_DIR, "inputs.npy"),
    inputs
)

np.save(
    os.path.join(SAVE_DIR, "targets.npy"),
    targets
)

print("\\nDataset Saved Successfully.")
