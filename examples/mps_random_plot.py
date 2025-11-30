import numpy as np

n = int(input("number of sites: "))

gammas = [0.1, 0.2, 0.4, 0.8]

# parameter setup
d = 2
bond_dim = 2**(n//2)
datasets = []
for j in range(4):
    dataset = []
    data = np.load("mps_random_plot_{}.npz".format(j+1))
    dataset.append(data["orig_expec"])
    dataset.append(data["ks"])
    dataset.append(data["rtrunc_means"])
    dataset.append(data["rtrunc_stds"])
    dataset.append(data["dtrunc_expecs"])
    dataset.append(gammas[j])
    datasets.append(dataset)


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

plt.rcParams['text.usetex'] = True
plt.rc('font', size=18)

# --- Create a 2x2 grid ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)

# Enforce identical left margins for ALL subplots →
# this prevents size changes from long/short y-tick labels
fig.subplots_adjust(left=0.17, right=0.97, bottom=0.10, top=0.93,
                    wspace=0.25, hspace=0.30)

# Flatten axes for easy iteration
axs = axs.ravel()

for ax, (orig_expec, ks_data, rmean, rstd, dexpec, gamma_val) in zip(axs, datasets):

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(np.arange(2, bond_dim+1, 4))
    ax.set_xlabel("bond dim. cutoff")
    ax.set_ylabel(r"$\langle Z_{5}Z_{6}\rangle$ rel. error")

    # rtrunc with error bars
    ax.errorbar(
        ks_data,
        (rmean - orig_expec) / abs(orig_expec),
        yerr=rstd / abs(orig_expec),
        fmt='o',
        capsize=4,
        color='blue',
        label="rtrunc (TD)"
    )

    # dtrunc
    ax.plot(
        ks_data,
        (dexpec - orig_expec) / abs(orig_expec),
        's',
        color='red',
        label='dtrunc'
    )

    # zero line
    ax.plot(ks_data, np.zeros_like(ks_data), '--', color='black')

    ax.set_title(r"$\gamma={}$".format(gamma_val))
    ax.legend()


fig.tight_layout()
plt.savefig("plots_2x2_grid.pdf")
#plt.show()