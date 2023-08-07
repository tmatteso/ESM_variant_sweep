import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from classifiers import *

def read_and_split(RECT_DF_LOCATION_ls, DELTA_EMBEDDINGS_ls, VAE_EMBEDDINGS_ls, UMAP_EMBEDDINGS_ls):
    clinvar_X, clinvar_y = [], [] 
    AR_X, AR_y = [], []
    AD_X, AD_y = [], []
    ADAR_X, ADAR_y = [], []
    X_and_ys = [clinvar_X, clinvar_y, AR_X, AR_y, AD_X, AD_y, ADAR_X, ADAR_y]
    for i in range(len(RECT_DF_LOCATION_ls)):
        #print(RECT_DF_LOCATION_ls[i])
        # load final df
        rectified_df = pd.read_csv(RECT_DF_LOCATION_ls[i])
        # load delta embeddings
        delta_embeds = np.load(DELTA_EMBEDDINGS_ls[i], allow_pickle=True)
        # load VAE Embeddings
        vae_embeds = np.load(VAE_EMBEDDINGS_ls[i], allow_pickle=True)
        # load UMAP Embeddings
        umap_embeds = np.load(UMAP_EMBEDDINGS_ls[i], allow_pickle=True)
        # put them together on index
        rectified_df["repr"] = delta_embeds.tolist()
        rectified_df["VAE_embed"] = vae_embeds.tolist()
        rectified_df["umap_embed"] = umap_embeds.tolist()
        # now that you have the rect df filled, you need to partition it based on classes to make the multiclass make sense
        available_labels = rectified_df.clinvar_label.unique()
        path_name_ls = ['Benign', 'Pathogenic', 'AR-GOF', 'AR-LOF', 'AD-LOF', 'AD-GOF', 'ADAR-LOF', 'ADAR-GOF']
        for i in range(0, len(path_name_ls), 2):
            if i in available_labels or i+1 in available_labels:
                X = rectified_df[(rectified_df["clinvar_label"] == i) | (rectified_df["clinvar_label"] == i+1)]
                y = X["clinvar_label"]
                X_and_ys[i].append(X), X_and_ys[i+1].append(y)    
    # with the split data, compute all metrics
    X_and_ys = [pd.concat(X_and_ys[i]) for i in range(len(X_and_ys))]
    # LLR ROC-AUC
    print("LLR ROC-AUC:", str(roc_auc_score(X_and_ys[1].to_numpy(), X_and_ys[0]["LLR"].to_numpy())))
    # knn LLR ROC-AUC
    X = X_and_ys[0]["LLR"].to_numpy()#.reshape(X_and_ys[0]["LLR"].to_numpy().shape[0], 1)
    y = X_and_ys[1].to_numpy()
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # all of these need to have a new arg split at the end with the train test split
    print("LLR-KNN ROC-AUC:",  special_knn(None, y, split=(X_train.reshape(X_train.shape[0], 1), y_train, 
                                                              X_test.reshape(X_test.shape[0], 1), y_test )))
    # define X and y for the esm roc-auc calc
    X = np.stack(X_and_ys[0].repr.values) #.reshape(X_and_ys[0]["LLR"].to_numpy().shape[0], 1)
    y = X_and_ys[1].to_numpy()
    print(X.shape, y.shape)
    print(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    multi_class='ovr'# do not actually compute ovr
    average="mean"
    # just run it dry with clinvar first
    
    print("roc-auc score", get_all_roc(X, y, available_labels, "ESM", multi_class, average, split=(X_train, y_train, X_test, y_test)))
    raise error
    one_gene_clf(X, y, available_labels, "ESM", multi_class,average)

    # define X and y for each LOF-GOF combo
    for i in range(2, len(path_name_ls), 2):
        X = np.stack(X_and_ys[i].repr.values) #.reshape(X_and_ys[0]["LLR"].to_numpy().shape[0], 1)
        y = X_and_ys[i+1].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        print(X.shape, y.shape)
        # computes knn for esm first
        get_all_roc(X, y, available_labels, "ESM")
        # computes all other classifiers for esm
        one_gene_clf(X, y, available_labels, "ESM")
    # then compute roc-auc from VAE for clinvar only

    # then compute roc-auc from UMAP for clinvar only

def main():
    name_ls = [
        "P42224", "Q8NBP7", "P07949", "Q14654", "P00441", "P63252", "Q14524", "P51787", "Q09428",
        "P41180", "P04275", "P29033", "Q12809", "P16473", "O43526", "Q01959", "P40763", "P35498"
            ]
    DELTA_EMBEDDINGS_ls = [name + "_delta_embeddings.npy" for name in name_ls]     
    RECT_DF_LOCATION_ls = [name + "_final_df.csv" for name in name_ls]   
    VAE_EMBEDDINGS_ls  =  [name + "_umap_embeddings.npy" for name in name_ls]  
    UMAP_EMBEDDINGS_ls =  [name + "_vae_embeddings.npy" for name in name_ls]
    read_and_split(RECT_DF_LOCATION_ls, DELTA_EMBEDDINGS_ls, VAE_EMBEDDINGS_ls, UMAP_EMBEDDINGS_ls)
    pass

if __name__ == "__main__":
     main()
