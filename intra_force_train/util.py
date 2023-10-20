import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

def plot_parity(y_true, y_pred,fname, y_pred_unc=None):
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)
    s_size=12
    l_size=16
    m_size=20
    b_width=1.5
    plt.rc('font',family='arial',size=l_size)
    plt.rc('axes',labelsize=m_size)
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'sans'
    mpl.rcParams['mathtext.it'] = 'sans:italic'
    mpl.rcParams['mathtext.default'] = 'it'
    #axmin = min(min(y_true), min(y_pred)) - 0.1*(max(y_true)-min(y_true))
    #axmax = max(max(y_true), max(y_pred)) + 0.1*(max(y_true)-min(y_true))
    axmin=min(y_pred)-0.1*min(y_pred)
    axmax=max(y_pred)+0.1*max(y_pred)
    axmin=-50
    axmax=50
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    fig=plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    plt.plot([axmin, axmax], [axmin, axmax], '--k')
    density = ax.scatter_density(y_true, y_pred, cmap=white_viridis)
    fig.colorbar(density, label='Points per pixel')
    
    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))
    
    ax = plt.gca()
    ax.set_aspect('equal')
    
    at = AnchoredText(
    f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}", prop=dict(size=l_size), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    
    plt.xlabel(r'$F_\mathrm{true}$ (kcal/mol/$\mathrm{\AA}$)')
    plt.ylabel(r'$F_\mathrm{GNN}$ (kcal/mol/$\mathrm{\AA}$)')
    
    plt.savefig(fname+'.png',dpi=300,bbox_inches="tight")
    plt.xlim((-35, 35))
    plt.ylim((-35, 35))
    plt.savefig(fname+'_adj.png',dpi=300,bbox_inches="tight")
    plt.close()
    return  