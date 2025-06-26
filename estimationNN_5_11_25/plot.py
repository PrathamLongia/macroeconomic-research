import os
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator
import matplotlib.pyplot as plt
from train_nn import Train_NN
import re


def plot_loss(ct: Train_NN):
    loss_path = os.path.join(ct.path, 'output_loss_S.txt')
    loss_values = []
    with open(loss_path, 'r') as file:
        for line in file:
            match = re.search(r'Loss = ([\d.e+-]+)', line)
            if match:
                loss = float(match.group(1))
                loss_values.append(loss)
                
    x_values = np.arange(1, len(loss_values) + 1)

    fig, ax = plt.subplots(figsize=(24, 12))
    ax.plot(x_values / 10, loss_values, linestyle='-')
    ax.set_xlabel('Epoch ($\\times$ 1000)', fontsize=20)
    ax.set_ylabel('Loss', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="-")
    # Increase the number of ticks on the y-axis
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))

    # Force scientific notation using FuncFormatter
    def scientific_notation(x, pos):
        return f"{x:.1e}"  # Format as scientific notation (e.g., 1.0e-5)

    ax.yaxis.set_major_formatter(FuncFormatter(scientific_notation))

    fig.savefig(ct.path + "/loss_S.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_ergodic_g_S_alpha(ct: Train_NN):
    g_erg = np.load('g_erg.npy')
    alpha_erg = np.load('alpha_erg.npy')
    S_erg = np.load('S_erg.npy') #-1/ct.par.xi* np.log(1/alpha_erg -1)

    # Plot ergodic g
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(g_erg, extent=[0, 1, 1, 0])  # (nx,ny)->(ny,nx)
    ax.set_ylabel("worker type $x$")
    ax.set_xlabel("firm type $y$")
    plt.colorbar(im)
    plt.title('Ergodic g')
    ax.invert_yaxis()
    filename = 'ergodic_g.png'
    savepath = os.path.join(ct.path, filename)
    plt.savefig(savepath)
    plt.close()

    # Plot Surplus and Alpha at ergodic g
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    fig, ((ax1, ax2)) = plt.subplots(2, 1)
    fig.set_size_inches(13, 13)
    fig.suptitle("Surplus and $\\alpha$ at Ergodic $g$")

    ax1.imshow(S_erg, extent=[0, 1, 1, 0])
    ax1.set_ylabel("worker type $x$")
    ax1.set_xlabel("firm type $y$")
    ax1.set_title("$S$ (at ergodic g)")
    plt.colorbar(ax1.imshow(S_erg, extent=[0, 1, 1, 0]), ax=ax1)
    ax1.invert_yaxis()

    ax2.imshow(alpha_erg, extent=[0, 1, 1, 0])
    ax2.set_ylabel("worker type $x$")
    ax2.set_xlabel("firm type $y$")
    ax2.set_title("$\\alpha$ (at ergodic g)")
    plt.colorbar(ax2.imshow(alpha_erg, extent=[0, 1, 1, 0]), ax=ax2)
    ax2.invert_yaxis()

    filename = 'S_alpha_ergodic.png'
    savepath = os.path.join(ct.path, filename)
    plt.savefig(savepath)
    plt.close()


def to_percent(y):
    return f"{y * 100:.2f}%"

def latex_label(label):
    special_symbols = {
        'beta': '\\beta',
        'gamma': '\\gamma',
        'alpha': '\\alpha',
        'delta': '\\delta',
        'rho': '\\rho',
        'zeta': '\\zeta',
        'eta': '\\eta',
        'theta': '\\theta',
        'iota': '\\iota',
        'kappa': '\\kappa',
        'lambda': '\\lambda',
        'mu': '\\mu',
        'nu': '\\nu',
        'xi': '\\xi',
        'omicron': '\\omicron',
        'pi': '\\pi',
        'sigma': '\\sigma',
        'tau': '\\tau',
        'upsilon': '\\upsilon',
        'phi': '\\phi',
        'chi': '\\chi',
        'psi': '\\psi',
        'omega': '\\omega',
        'Gamma': '\\Gamma',
        'Delta': '\\Delta',
        'Theta': '\\Theta',
        'Lambda': '\\Lambda',
        'Xi': '\\Xi',
        'Pi': '\\Pi',
        'Sigma': '\\Sigma',
        'Upsilon': '\\Upsilon',
        'Phi': '\\Phi',
        'Psi': '\\Psi',
        'Omega': '\\Omega',
        'epsilon': '\\epsilon',
        'vartheta': '\\vartheta',
        'varpi': '\\varpi',
        'varrho': '\\varrho',
        'varsigma': '\\varsigma',
        'varphi': '\\varphi',
        'lam_L': '\\lambda_L',
        'lam_H': '\\lambda_H',
    }
    if label in special_symbols:
        return f"${special_symbols[label]}$"
    else:
        return f"${label}$"