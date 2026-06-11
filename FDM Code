import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# PARAMETERS
# ==========================================================

lambda1 = 0.01

alpha = 2.0
l = 0.5

delta1 = 0.2
delta0 = 0.1

A = 1.0

# ==========================================================
# DOMAIN
# ==========================================================

L = 1.0
T = 1.0

Nx = 201
dx = L/(Nx-1)

x = np.linspace(0,L,Nx)

# Explicit stability condition

dt = 0.4*dx**2/lambda1

Nt = int(T/dt)

print("dx =",dx)
print("dt =",dt)
print("Nt =",Nt)

# ==========================================================
# INITIAL CONDITIONS
# ==========================================================

S = np.ones(Nx)*5.0

C = 2*np.exp(-50*(x-0.5)**2)

P = np.zeros(Nx)

# ==========================================================
# STORAGE
# ==========================================================

save_every = max(1,Nt//200)

S_hist = []
C_hist = []
P_hist = []

time_hist = []

# ==========================================================
# TIME STEPPING
# ==========================================================

for n in range(Nt):

    t = n*dt

    S_old = S.copy()
    C_old = C.copy()
    P_old = P.copy()

    # ======================================================
    # NEUMANN BC
    # dS/dx = dC/dx = dP/dx = 0
    # ======================================================

    S_old[0] = S_old[1]
    S_old[-1] = S_old[-2]

    C_old[0] = C_old[1]
    C_old[-1] = C_old[-2]

    P_old[0] = P_old[1]
    P_old[-1] = P_old[-2]

    # ======================================================
    # INTERIOR UPDATE
    # ======================================================

    for i in range(1,Nx-1):

        S_xx = (
            S_old[i+1]
            -2*S_old[i]
            +S_old[i-1]
        )/dx**2

        C_xx = (
            C_old[i+1]
            -2*C_old[i]
            +C_old[i-1]
        )/dx**2

        P_xx = (
            P_old[i+1]
            -2*P_old[i]
            +P_old[i-1]
        )/dx**2

        # -----------------------------------------------
        # Reaction terms
        # -----------------------------------------------

        RS = (
            A
            - alpha*S_old[i]*C_old[i]
            + delta1*P_old[i]
            - delta0*S_old[i]
        )

        RC = (
            alpha*S_old[i]*C_old[i]
            - l*C_old[i]
            - delta0*C_old[i]
        )

        RP = (
            l*C_old[i]
            - delta1*P_old[i]
            - delta0*P_old[i]
        )

        # -----------------------------------------------
        # Explicit Euler
        # -----------------------------------------------

        S[i] = (
            S_old[i]
            + dt*(lambda1*S_xx + RS)
        )

        C[i] = (
            C_old[i]
            + dt*(lambda1*C_xx + RC)
        )

        P[i] = (
            P_old[i]
            + dt*(lambda1*P_xx + RP)
        )

    # ======================================================
    # SAVE
    # ======================================================

    if n % save_every == 0:

        S_hist.append(S.copy())
        C_hist.append(C.copy())
        P_hist.append(P.copy())

        time_hist.append(t)

# ==========================================================
# ARRAYS
# ==========================================================

S_hist = np.array(S_hist)
C_hist = np.array(C_hist)
P_hist = np.array(P_hist)

time_hist = np.array(time_hist)

# ==========================================================
# HEATMAPS
# ==========================================================

fig,ax = plt.subplots(1,3,figsize=(18,5))

im1 = ax[0].imshow(
    S_hist,
    extent=[0,L,0,T],
    origin='lower',
    aspect='auto',
    cmap='viridis'
)

ax[0].set_title('Susceptible Population')
ax[0].set_xlabel('Space')
ax[0].set_ylabel('Time')

plt.colorbar(im1,ax=ax[0])

im2 = ax[1].imshow(
    C_hist,
    extent=[0,L,0,T],
    origin='lower',
    aspect='auto',
    cmap='viridis'
)

ax[1].set_title('Criminal Population')
ax[1].set_xlabel('Space')
ax[1].set_ylabel('Time')

plt.colorbar(im2,ax=ax[1])

im3 = ax[2].imshow(
    P_hist,
    extent=[0,L,0,T],
    origin='lower',
    aspect='auto',
    cmap='viridis'
)

ax[2].set_title('Prison Population')
ax[2].set_xlabel('Space')
ax[2].set_ylabel('Time')

plt.colorbar(im3,ax=ax[2])

plt.tight_layout()
plt.show()

# ==========================================================
# SNAPSHOTS
# ==========================================================

indices = [
    0,
    len(time_hist)//3,
    2*len(time_hist)//3,
    -1
]

fig,ax = plt.subplots(1,4,figsize=(18,4))

for k,idx in enumerate(indices):

    ax[k].plot(
        x,
        C_hist[idx],
        linewidth=2
    )

    ax[k].set_title(
        f"t = {time_hist[idx]:.3f}"
    )

    ax[k].grid(True)

    ax[k].set_xlabel("x")
    ax[k].set_ylabel("C(x,t)")

plt.tight_layout()
plt.show()
