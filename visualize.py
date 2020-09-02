import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from IPython.display import display





def display_model_comparison(comp_models, 
                             show_roc=False, 
                             show_cm=True, 
                             show_pov_rate_error=False, 
                             highlight_best=True, 
                             transpose=False, 
                             rank_order=False):
    # Don't highlight if only showing one model
    if len(comp_models) == 1:
        highlight_best = False

    # Plot ROC and confusion matrix in a single row
    n_axes = sum([show_roc, show_cm])
    if n_axes > 0:
        fig, axes = plt.subplots(1,n_axes, figsize=(5*n_axes,4))
        i = 0
        
        if n_axes == 1:
            axes = [axes]

        # Plot ROC curve
        if show_roc:
            ax = axes[i]
            plot_roc(comp_models, ax)
            i += 1

        # Plot confusion matrices
        if show_cm:
            ax = axes[i]
            conf_mat = comp_models[0]['confusion_matrix']
            plot_confusion_matrix(conf_mat, 
                                  classes=['Non-poor', 'Poor'], 
                                  reverse=True,
                                  normalize=True,
                                  ax=ax, 
                                  fig=fig)

        fig.tight_layout()
        plt.show()

    # Display score table
    disp_metrics = ['accuracy', 'recall', 'precision', 'f1', 'cross_entropy', 'roc_auc', 'cohen_kappa']
    disp_types = {'max': ['accuracy', 'recall', 'precision', 'f1', 'roc_auc', 'cohen_kappa'], 
                  'min': ['cross_entropy']}
    if show_pov_rate_error == True:
        disp_metrics.append('pov_rate_error')
        disp_types['abs_min'] = ['pov_rate_error']
    
    met_df = []
    for m in comp_models:
        m_disp = {x[0]: x[1] for x in m.items() if x[0] in disp_metrics}
        met_df.append(pd.DataFrame.from_dict(m_disp, orient='index')
                      .rename(columns={0:m['name']})
                      .loc[disp_metrics])
    met_df = pd.concat(met_df, axis=1)

    if transpose:
        met_df = met_df.T
        axis = 0
        highlight_index = {'max': pd.IndexSlice[:, disp_types['max']], 
                           'min': pd.IndexSlice[:, disp_types['min']]}
        if show_pov_rate_error == True:
            highlight_index['abs_min'] = pd.IndexSlice[:, disp_types['abs_min']]
        
        if rank_order:
            met_df['mean_rank'] = met_df.apply(get_rank_order).mean(axis=1)
            met_df = met_df.sort_values('mean_rank')
            
    else:
        axis = 1
        highlight_index = {'max': pd.IndexSlice[disp_types['max'], :], 
                           'min': pd.IndexSlice[disp_types['min'], :]}
        if show_pov_rate_error == True:
            highlight_index['abs_min'] = pd.IndexSlice[disp_types['abs_min'], :]


    scores = met_df.style
    if highlight_best:
        scores = scores.highlight_max(subset=highlight_index['max'], 
                                      color='steelblue',
                                      axis=axis)
        scores = scores.highlight_min(subset=highlight_index['min'], 
                                      color='steelblue',
                                      axis=axis)
        if show_pov_rate_error == True:
            scores = scores.apply(highlight_abs_min, 
                                  subset=highlight_index['abs_min'], 
                                  color='steelblue',
                                  axis=axis)

    display(scores.set_caption("Model Scores"))
    return met_df

def display_precision_recall(results):
    if len(results > 10):
        fig, ax = plt.subplots(figsize=(6,len(results)*0.3))
    else:
        fig, ax = plt.subplots()
    (results.sort_values('mean_rank', ascending=False)
     [['recall', 'precision']]
     .plot.barh(title='Precision and Recall', ax=ax))
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles[::-1], labels=labels[::-1], bbox_to_anchor=(1.3, 1))
    plt.show()

def display_feat_ranks(feats):
    feat_rankings = []
    for key, value in sorted(feats.items()):
        if type(value) == pd.DataFrame:
            if 'abs' in value.columns:
                ranks = (pd.DataFrame(value['abs'].rank(ascending=False).rename(key)))
            if 'importance' in value.columns:
                ranks = (pd.DataFrame(value['importance'].rank(ascending=False).rename(key)))
            feat_rankings.append(ranks)
    feat_rankings = pd.concat(feat_rankings, axis=1)
    mean_rank = feat_rankings.mean(axis=1)
    counts = feat_rankings.count(axis=1)
    feat_rankings['mean_rank'] = mean_rank
    feat_rankings['count'] = counts
    feat_rankings = feat_rankings.sort_values('mean_rank', ascending=True)

    display(feat_rankings.style.set_caption("Feature Ranking"))
    return feat_rankings
