import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
# I want you to build a CI pipeline for this with Github Actions
# use the shell scripts you made

# move the knn_and roc code here
def knn_and_roc(X, y, multi_class='raise', average="macro", classifier=None): #KNeighborsClassifier(n_neighbors=5)): 
    # raise means it will complain about multiclass until overridden
    pred_ls = []
    # calc for each var in the X
    for data_i in range(X.shape[0]):
        # Classify that variants with KNN (e.g. for K=5), i.e. assign each variant with a score 0<=s<=1 
        # which indicates the fraction of its K closest variants that are pathogenic (excluding the variant itself of course).
        if classifier != None:
            #print(classifier)
            knn = classifier_space[classifier]
        else:
            knn = KNeighborsClassifier(n_neighbors=5) 
        # need to concat
        train_X, train_y = np.concatenate([X[:data_i], X[data_i+1:]]), np.concatenate([y[:data_i], y[data_i+1:]])
        #print(y[data_i])
        test_X, test_y =  X[data_i].reshape(1, X.shape[1]), y[data_i].reshape(1, 1)
        # make sure that the variant you want is excluded in train, and only that var in test
        # count the number of instances of each class, if anyone is 1, return "NaN"
        # Get unique elements and their counts
        unique_elements, counts = np.unique(train_y, return_counts=True)
        if 1 in counts:
            return "NaN"
        knn.fit(train_X, train_y) # not using anything held out
        # Each variant should now have a true label (y_true) and predicted label probably (y_pred_prob). 
        # Use the ROC-AUC metric to determine to what extent y_pred_prob is a good predictor of y_true. 
        # This will give you a number between 0 and 1 which indicates whether variants of the same label 
        # tend to cluster together (but without having to run any clustering algorithm).
        if multi_class=='ovr':
            pred_ls.append(knn.predict_proba(test_X))
        else:
            pred_ls.append(knn.predict(test_X))
    try:
        pred_ls = np.array(pred_ls) 
    except:
        print("There is only one example of a pathogenicity class")
        raise ValueError
    if multi_class=='ovr':
        pred_ls = pred_ls.reshape(pred_ls.shape[0], pred_ls.shape[-1])
    return roc_auc_score(y, pred_ls, multi_class=multi_class, average=average) 
 
def get_all_roc(X, y, path_name_in_df, roc_type, multi_class, average, classifier=None): #KNeighborsClassifier(n_neighbors=5)):    
    path_name_ls = ['Benign', 'Pathogenic', 'AR-GOF', 'AR-LOF', 'AD-LOF', 'AD-GOF', 'ADAR-LOF', 'ADAR-GOF']
    k = 0
    if multi_class == "raise":
        print(roc_type +" ROC-AUC: " + str(knn_and_roc(X, y, classifier=classifier)))       
    else:
        # need to modify knn_and_roc behavior for multiclass
        # partition into clinvar part
        if 0 in path_name_in_df and 1 in path_name_in_df:
            clinvar_X = X[(y == 0) | (y == 1)]
            clinvar_y = y[(y == 0) | (y == 1)]
            if classifier == None:
                print(1)
                print(roc_type  + " kNN Clinvar ROC-AUC: " +  str(knn_and_roc(clinvar_X, clinvar_y))) #, classifier=KNeighborsClassifier(n_neighbors=5))))
            else:
                print(y)
                print(roc_type, clf_names[classifier], "Clinvar ROC-AUC:",  str(knn_and_roc(clinvar_X, clinvar_y, classifier=classifier)))
        else:
            print(roc_type  + " Clinvar ROC-AUC: NaN")
        # partition into LOF-GOF part
        label_num = max(path_name_in_df[~np.isnan(path_name_in_df)])
        lof_gof_X = X[(y == label_num - 1) | (y == label_num)] 
        lof_gof_y = y[(y == label_num - 1) | (y == label_num)]
        # old version of multiclass #roc_auc_ls = knn_and_roc(X, y, multi_class, average)
        #for i in range(len(path_name_ls)):
        #    if i in path_name_in_df:
        if classifier == None:
            print(3)
            print(roc_type, "kNN LOF-GOF ROC-AUC:",  str(knn_and_roc(lof_gof_X, lof_gof_y, classifier=classifier)))
        else:
            print(4)
            print(roc_type, clf_names[classifier], "LOF-GOF ROC-AUC:",  str(knn_and_roc(lof_gof_X, lof_gof_y, classifier=classifier)))
        # path_name_ls[i] +" is "+ str(roc_auc_ls[k]))
 
# run G-NB, RF, LR; return labels and compute ROC_AUC later
global classifier_space 
classifier_space = [GaussianNB(), RandomForestClassifier(), LogisticRegression()]
global clf_names 
clf_names = ["Gaussian NB", "Random Forest", "Logistic Regression"]

def one_gene_clf(X, y, path_name_in_df, roc_type, multi_class,average):
    # one gene train/test -- use the knn_and_roc function
    # do wrt to both label sets -- needs to be passed in
    results = []
    # I think we should just pass around the clf index, rather than the object itself
    for i in range(len(classifier_space)):
        # make sure the print statement says which classifier it is
        get_all_roc(X, y, path_name_in_df, roc_type, multi_class, average, classifier=i)
        

# all gene train/test
def all_gene_clf(X, y, path_name_in_df, gene_space,multi_class,average):
    full_X_train, full_X_test, full_y_train, full_y_test = [], [], [], []
    for gene in gene_space:
        # do a 75-25 split wrt to variants
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        # put all genes together
        full_X_train += X_train
        full_X_test += X_test
        full_y_train += y_train
        full_y_test += y_test
    # for each classifier, .fit(X,y)
    for clf in classifier_space:
        clf.fit(full_X_train,full_y_train)
        # pred the y
        y_pred = knn.predict(full_X_test)
        # calc roc
        X, y = full_X_test, y_pred
        path_name_ls = ['Benign', 'Pathogenic', 'AR-GOF', 'AR-LOF', 'AD-LOF', 'AD-GOF', 'ADAR-LOF', 'ADAR-GOF']
        # print clinvar
        if 0 in path_name_in_df and 1 in path_name_in_df:
            clinvar_X = X[(y == 0) | (y == 1)]
            clinvar_y = y[(y == 0) | (y == 1)]
            print(roc_type, "Clinvar", clf_name_i, "ROC-AUC:", roc_auc_score(clinvar_X, clinvar_y, multi_class=multi_class, average=average))
        else:
            print(roc_type, "Clinvar", clf_name_i, "ROC-AUC: NaN")
        # partition into LOF-GOF part
        label_num = max(path_name_in_df[~np.isnan(path_name_in_df)])
        lof_gof_X = X[(y == label_num - 1) | (y == label_num)]
        lof_gof_y = y[(y == label_num - 1) | (y == label_num)]
        print(roc_type, "LOF-GOF", clf_name_i, " ROC-AUC:", roc_auc_score(lof_gof_X, lof_gof_y, multi_class=multi_class, average=average))



# for multiclass and results
#       if multi_class=='ovr':
#            pred_ls.append(knn.predict_proba(test_X))
#        else:
#            pred_ls.append(knn.predict(test_X))
#    try:
#        pred_ls = np.array(pred_ls)
#    except:
#        print("There is only one example of a pathogenicity class")
#        raise ValueError
#    if multi_class=='ovr':
#        pred_ls = pred_ls.reshape(pred_ls.shape[0], pred_ls.shape[-1])
#    return roc_auc_score(y, pred_ls, multi_class=multi_class, average=average)# 

