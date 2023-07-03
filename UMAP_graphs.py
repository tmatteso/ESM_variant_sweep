import matplotlib.pyplot as plt
import pandas as pd

# probably need to add sabve fig

def get_LLR_graph(rectified_df, GENE_NAME, get_pos=True):
    if get_pos:
        rectified_df["pos"] = (rectified_df["mutation_name"].str[1:-1]).astype(int)
    # overlay the LLR with UMAP
    WT_STYLES = [
        dict(c = 'black', s = 200, marker = 's'),
        dict(c = 'white', s = 100, marker = 's'),
        dict(c = 'green', s = 75, label = 'WT'),
    ]
    if get_pos:
        COLOR_SCHEMES = [
            # color_field, cmap
            ('pos', 'Reds_r'),
            ('LLR', 'Reds_r'),
            #('dist_from_WT', 'Reds'),
        ]
    else:
        COLOR_SCHEMES = [
            # color_field, cmap
            ('LLR', 'Reds_r'),]
    # all mutations -> new_mutations
    new_mutations = rectified_df.copy()
    #(_, WT_mutation), = new_mutations[new_mutations['mutation_name'] == 'WT'].iterrows()

    for color_field, cmap in COLOR_SCHEMES:

        fig, ax = plt.subplots(figsize = (12, 8))
        ax.set_aspect('equal', 'datalim') # was ' & '.join(new_mutations['gene_name'].unique()) before GENE_NAME
        ax.set_title('%s (colored by %s)' % ((GENE_NAME), color_field))

        relevant_mutations = new_mutations.dropna(subset = ['norm_LLR'])
        scatter_plot = ax.scatter(relevant_mutations['umap1'], relevant_mutations['umap2'], \
                c = relevant_mutations[color_field], cmap = cmap, s = 3, alpha = 0.5) # s = 3, alpha = 0.5
        colorbar = fig.colorbar(scatter_plot)
        colorbar.set_label(color_field, fontsize = 14)

    #     for wt_style in WT_STYLES:
    #         plt.scatter(WT_mutation['umap1'], WT_mutation['umap2'], **wt_style)
        ax.legend(fontsize = 16)
        fig.savefig(GENE_NAME+ '_' + color_field +'.png')

def get_clinvar_graph(GENE_NAME, new_mutations):    
    # and make a Clinvar plot -- send ASAP
    fig, ax = plt.subplots(figsize = (12, 8))
    # all mutations -> new_mutations

    #GENE_NAME = "PKD1"
    ax.set_title('%s variants (colored by ClinVar annotations)' % GENE_NAME, fontsize = 18)

    ax.scatter(new_mutations['umap1'], new_mutations['umap2'], s = 3, color = '#E5E7E9')
    # now a for loop from new_mutations['path_name']
    path_name_ls = ['Benign', 'Pathogenic', 'AR-GOF', 'AR-LOF', 'AD-LOF', 'AD-GOF', 'ADAR-LOF', 'ADAR-GOF']
    #color_ls = "blue", "red", "yellow", "green", "black
    for i in range(len(path_name_ls)):
        if sum(new_mutations['clinvar_label'] == i) != 0:
            ax.scatter(new_mutations.loc[(new_mutations['clinvar_label'] == i), 'umap1'], 
               new_mutations.loc[(new_mutations['clinvar_label'] == i),
            'umap2'], s = 50, label = path_name_ls[i], alpha =0.5)
#     ax.scatter(new_mutations.loc[new_mutations['clinvar_label'] == 1, 'umap1'], new_mutations.loc[new_mutations['clinvar_label'] == 1, \
#             'umap2'], s = 50, color = 'red', label = 'Pathogenic', alpha =0.5) #color = '#CB4335'
#     ax.scatter(new_mutations.loc[(new_mutations['clinvar_label'] == 0), 'umap1'], 
#                new_mutations.loc[(new_mutations['clinvar_label'] == 0),
#             'umap2'], s = 50, color = 'blue', label = 'Benign', alpha =0.5) # color = '#1A5276'
    # for wt_style in WT_STYLES:
    #     plt.scatter(WT_mutation['umap1'], WT_mutation['umap2'], **wt_style)
    ax.legend(fontsize = 16)
    fig.savefig(GENE_NAME + '_clinvar.png')