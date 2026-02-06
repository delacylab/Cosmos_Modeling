import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from .knee_identifier import knee

def ablate(df_shap: pd.DataFrame,
           shap_col: str,     
           out_path: str):
    df_shap = df_shap.sort_values(by=shap_col, ascending=False)
    feats: list[str] = df_shap['Feature'].to_list()
    mashap: list[float] = df_shap[shap_col].to_list()
    elbow_x, feat_order = knee(mashap)
    elbow_y = mashap[elbow_x]
    
    # Identify the ablated feature subset
    feats_ablated: list[str] = df_shap.iloc[:elbow_x+1]['Feature'].to_list()

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(df_shap.shape[0]), mashap, 
            label='Ordered mean absolute SHAP values')
    ax.plot(elbow_x, elbow_y, markersize=10, marker='D', color='tab:red', linestyle='None', 
            label=f'Point of diminishing returns ({elbow_x+1} predictors)')
    ax.grid(alpha=0.2)
    ax.set_xticks(range(0, df_shap.shape[0], 50))
    ax.set_xticklabels(labels=ax.get_xticks(), fontsize=14)
    ax.set_xlim([-1, len(mashap)+1])
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('Ordered predictor index', fontsize=16)
    ax.set_ylabel('Mean predictor importance', fontsize=16)
    ax.legend(loc='upper right', fontsize=13)
    if out_path is not None:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'Image saved with {len(feats_ablated)} ablated features.')
    plt.close(fig)
    return feats_ablated
