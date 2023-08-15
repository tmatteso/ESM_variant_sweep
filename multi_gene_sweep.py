import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from classifiers import *
from sklearn.model_selection import ShuffleSplit

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
    return X_and_ys

def compute_metrics(X_and_ys):
    path_name_ls = ['Benign', 'Pathogenic', 'AR-GOF', 'AR-LOF', 'AD-LOF', 'AD-GOF', 'ADAR-LOF', 'ADAR-GOF']
    # these need to be recomputed too based on the CV
    # LLR ROC-AUC
    # when the values are flipped, roc auc score gets mad
    #print("LLR ROC-AUC:", str(roc_auc_score(X_and_ys[1].to_numpy(), X_and_ys[0]["LLR"].to_numpy())))
    # knn LLR ROC-AUC
    #X = X_and_ys[0]["LLR"].to_numpy()#.reshape(X_and_ys[0]["LLR"].to_numpy().shape[0], 1)
    #y = X_and_ys[1].to_numpy()
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # all of these need to have a new arg split at the end with the train test split
    # replace this with new one
    #print("LLR-KNN ROC-AUC:", classifier_full_split(X, #X.reshape(X.shape[0], 1), 
    #                                                y, X_train.reshape(X_train.shape[0], 1), 
    #                                                X_test.reshape(X_test.shape[0], 1), y_train, y_test))
    #print("LLR-KNN ROC-AUC:",  special_knn(None, y, split=(X_train.reshape(X_train.shape[0], 1), y_train, 
    #                                                          X_test.reshape(X_test.shape[0], 1), y_test )))
    # define X and y for the esm roc-auc calc
    #X = np.stack(X_and_ys[0].repr.values) #.reshape(X_and_ys[0]["LLR"].to_numpy().shape[0], 1)
    #y = X_and_ys[1].to_numpy()
    embed_names = ["ESM", "VAE", "UMAP"]
    # define X and y for each LOF-GOF combo
    for i in range(0, len(path_name_ls), 2):
        #print(len(X_and_ys[i]))
        #print(len(X_and_ys[i+1]))
        LLR_X = X_and_ys[i]["LLR"].to_numpy()
        esm_X = np.stack(X_and_ys[i].repr.values) #.reshape(X_and_ys[0]["LLR"].to_numpy().shape[0], 1)
        vae_X = np.stack(X_and_ys[i]["VAE_embed"].values)
        umap_X = np.stack(X_and_ys[i]["umap_embed"].values)
        y = X_and_ys[i+1].to_numpy() 
        # now make this into 5 fold CV, then apply the same splits to the VAE and UMAP embeds
        cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
        llr, llr_knn = [], []
        kfold_knn = [[], [], []] # esm, vae, umap
        kfold_gnb = [[], [], []]
        kfold_rf = [[], [], []]
        kfold_lr = [[], [], []]
        for j, (train_index, test_index) in enumerate(cv.split(esm_X)):
            X_train = esm_X[train_index]
            X_test = esm_X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            label_name  = "ESM " +  path_name_ls[i]+ " " + path_name_ls[i+1]
            # LLR only
            llr.append(roc_auc_score(y_test, LLR_X[test_index]))
            # LLR knn
            llr_knn.append(classifier_full_split(LLR_X, y, 
                LLR_X[train_index].reshape(LLR_X[train_index].shape[0], 1), 
                LLR_X[test_index].reshape(LLR_X[test_index].shape[0], 1), 
                y_train, y_test))
            # ESM knn
            kfold_knn[0].append(classifier_full_split(esm_X, y, X_train, X_test, y_train, y_test))
            # now have ovr clf return an array of scores
            esm_other_clf =  over_clf_space(esm_X, y, "classifier_full_split", (X_train, X_test, y_train, y_test))
            kfold_gnb[0].append(esm_other_clf[0]), kfold_rf[0].append(esm_other_clf[1]), kfold_lr[0].append(esm_other_clf[2])
            # then compute roc-auc from VAE for each binary classification task
            X_train = vae_X[train_index]
            X_test = vae_X[test_index]
            kfold_knn[1].append(classifier_full_split(vae_X, y, X_train, X_test, y_train, y_test))
            vae_other_clf = over_clf_space(vae_X, y, "classifier_full_split", (X_train, X_test, y_train, y_test))
            kfold_gnb[1].append(vae_other_clf[0]), kfold_rf[1].append(vae_other_clf[1]), kfold_lr[1].append(vae_other_clf[2])
            # then compute roc-auc from UMAP for each binary classification task
            X_train = umap_X[train_index]
            X_test = umap_X[test_index]
            kfold_knn[2].append(classifier_full_split(umap_X, y, X_train, X_test, y_train, y_test))
            umap_other_clf = over_clf_space(umap_X, y, "classifier_full_split", (X_train, X_test, y_train, y_test))
            kfold_gnb[2].append(umap_other_clf[0]), kfold_rf[2].append(umap_other_clf[1]), kfold_lr[2].append(umap_other_clf[2])
        # print the knn avg after the kfold split for each embedding type
        print("LLR", path_name_ls[i]+ " " + path_name_ls[i+1], "avg ROC-AUC:", np.mean(llr))
        print("LLR-kNN", path_name_ls[i]+ " " + path_name_ls[i+1], "avg ROC-AUC:", np.mean(llr_knn))
        for k in range(len(embed_names)):
            label_name  = embed_names[k]+ " "+  path_name_ls[i]+ " " + path_name_ls[i+1]
            #print("LLR", path_name_ls[i]+ " " + path_name_ls[i+1], " avg ROC-AUC:", np.mean(llr[k]))
            #print("LLR-kNN", path_name_ls[i]+ " " + path_name_ls[i+1], " avg ROC-AUC:", np.mean(llr_knn[k]))
            print(label_name, "kNN avg ROC-AUC:", np.mean(kfold_knn[k]))#, "+/-", np.std(kfold_knn[k]))
            print(label_name, "Gaussian NB avg ROC-AUC:", np.mean(kfold_gnb[k]))#, "+/-", np.std(kfold_gnb[k]))
            print(label_name, "Random Forest avg ROC-AUC:", np.mean(kfold_rf[k]))#, "+/-", np.std(kfold_rf[k]))
            print(label_name, "Logistic Regression avg ROC-AUC:", np.mean(kfold_lr[k]))#, "+/-", np.std(kfold_lr[k])) 

def main():
    name_ls = [
        "P42224", "Q8NBP7", "P07949", "Q14654", "P00441", "P63252", "Q14524", "P51787", "Q09428",
        "P41180", "P04275", "P29033", "Q12809", "P16473", "O43526", "Q01959", "P40763", "P35498"
            ]
    DELTA_EMBEDDINGS_ls = [name + "_delta_embeddings.npy" for name in name_ls]     
    RECT_DF_LOCATION_ls = [name + "_final_df.csv" for name in name_ls]   
    VAE_EMBEDDINGS_ls  =  [name + "_umap_embeddings.npy" for name in name_ls]  
    UMAP_EMBEDDINGS_ls =  [name + "_vae_embeddings.npy" for name in name_ls]
    X_and_ys = read_and_split(RECT_DF_LOCATION_ls, DELTA_EMBEDDINGS_ls, VAE_EMBEDDINGS_ls, UMAP_EMBEDDINGS_ls)
    compute_metrics(X_and_ys)

if __name__ == "__main__":
     main()
