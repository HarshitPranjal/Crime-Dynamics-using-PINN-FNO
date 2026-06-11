# ==========================================================
# MULTI-TIME SNAPSHOT COMPARISON
# ==========================================================

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({

    "figure.dpi":300,
    "savefig.dpi":600,

    "font.family":"serif",
    "font.size":13,

    "axes.titlesize":14,
    "axes.labelsize":13,

    "savefig.bbox":"tight"

})

# ----------------------------------------------------------
# SAMPLE TO VISUALIZE
# ----------------------------------------------------------

# The 'sample' variable is no longer needed as 'snap' will now iterate through samples.
# sample = 0

# ----------------------------------------------------------
# CRIMINAL COMPARTMENT
# ----------------------------------------------------------

channel = 1 # Criminal population (C) is at index 1

# ----------------------------------------------------------
# FIVE SAMPLES FROM BATCH (instead of snapshots in time)
# ----------------------------------------------------------

# Assuming 'truth' and 'pred' are from the FNO model with shape (batch_size, channels, height, width)
# Let's pick 5 samples from the batch.
num_samples_to_plot = min(5, truth.shape[0]) # Use min to not exceed available batch size
snapshot_ids = list(range(num_samples_to_plot))

fig, axes = plt.subplots(

    len(snapshot_ids),
    3,

    figsize=(12,16),

    constrained_layout=True

)

# Ensure axes is always 2D for consistent indexing
if len(snapshot_ids) == 1:
    axes = axes[np.newaxis, :]

for row, snap_idx in enumerate(snapshot_ids):

    # Correct indexing to get a 2D image: truth[batch_idx, channel_idx, height, width]
    truth_img = truth[snap_idx, channel, :, :]

    pred_img = pred[snap_idx, channel, :, :]

    error_img = np.abs(
        truth_img - pred_img
    )

    vmin = min(
        truth_img.min(),
        pred_img.min()
    )

    vmax = max(
        truth_img.max(),
        pred_img.max()
    )

    # --------------------------------------------------
    # FDM Reference
    # --------------------------------------------------

    im0 = axes[row,0].imshow(

        truth_img,

        cmap="viridis",

        origin="lower",

        vmin=vmin,
        vmax=vmax

    )

    # --------------------------------------------------
    # FNO Prediction
    # --------------------------------------------------

    im1 = axes[row,1].imshow(

        pred_img,

        cmap="viridis",

        origin="lower",

        vmin=vmin,
        vmax=vmax

    )

    # --------------------------------------------------
    # ERROR
    # --------------------------------------------------

    im2 = axes[row,2].imshow(

        error_img,

        cmap="viridis",

        origin="lower"

    )

    axes[row,0].set_ylabel(

        f"Sample {snap_idx}", # Changed label to reflect sample index

        fontsize=13

    )

    if row == 0:

        axes[row,0].set_title(
            "FDM Reference"
        )

        axes[row,1].set_title(
            "FNO Prediction"
        )

        axes[row,2].set_title(
            "Absolute Error"
        )

# ----------------------------------------------------------
# COLORBARS
# ----------------------------------------------------------

# Ensure colorbars are correctly applied to all rows
fig.colorbar(
    im0,
    ax=axes[:,0],
    fraction=0.025
)

fig.colorbar(
    im1,
    ax=axes[:,1],
    fraction=0.025
)

fig.colorbar(
    im2,
    ax=axes[:,2],
    fraction=0.025
)

# ----------------------------------------------------------
# TITLE
# ----------------------------------------------------------

plt.savefig(

    "multi_snapshot_fdm_fno.pdf"

)

plt.savefig(

    "multi_snapshot_fdm_fno.png"

)

plt.show()
