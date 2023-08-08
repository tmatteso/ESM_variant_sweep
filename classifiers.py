import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

# then I think a refactor is in order for special knn and knn_and_roc
# the changes you made to make refactor work are pretty atrocious
# no more than nesting 2 loops or 2 conditionals at a time, and only 3 total if trying both
# need more subroutines, functions have become too large
# optimal function size for me is 10<=x=<50 lines





def special_knn_sep(class_weights, X_train, X_test, y_train, y_test):
    # this part is boiler plate for any knn
    if len(y_train) < 10:
        knn = NearestNeighbors(n_neighbors=len(y_train))
    else:
        knn = NearestNeighbors(n_neighbors=10)
    unique_elements, counts = np.unique(y_train, return_counts=True)
    knn.fit(X_train, y_train) # not using anything held out
    # now get the nearest neighbors out, based on the test_X query
    neighbors = knn.kneighbors(X_test, return_distance=False)
    # should be np array, check for shape here
    neighbor_classes = np.take(y_train, neighbors, 0) # final arg is axis number, check to make sure which is which
    # broadcast the multiplication with your class_weights
    class_votes = [class_weights[i] * len(neighbor_classes[neighbor_classes == unique_elements[i]]) for i in range(len(unique_elements))]
    return unique_elements[np.argmax(class_votes)]

def special_knn_2d(class_weights, X_train, X_test, y_train, y_test):
    knn = NearestNeighbors(n_neighbors=10)
    pred_ls = []
    unique_elements, counts = np.unique(y_train, return_counts=True)
    knn.fit(X_train, y_train) # not using anything held out 
    # now get the nearest neighbors out, based on the test_X query
    neighbors = knn.kneighbors(X_test, return_distance=False)
    # should be np array, check for shape here
    neighbor_classes = np.take(y_train, neighbors, 0)
    # for arr in 2d arr
    for j in range(neighbor_classes.shape[0]):
    # if arr has more 0 than 1, call as 0 , o.w. 1
        class_votes =[class_weights[i]*len(neighbor_classes[j][neighbor_classes[j] == unique_elements[i]]) for i in range(len(unique_elements))]
        # make the prediction based on the reweighted votes
        pred_ls.append(unique_elements[np.argmax(class_votes)])
    # do this for whole vect, returning a 2d arr.shape[0] vect
    return np.array(pred_ls)

def other_clf_sep(classifier, X_train, X_test, y_train, y_test):
    clf = classifier_space[classifier]
    unique_elements, counts = np.unique(y_train, return_counts=True)
    if 1 in counts and clf_names[classifier] == "Logistic Regression":
        return "NaN"
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

# classifier leave one out
def classifier_LOO(X,y, classifier=None):
    # calculate class proportions and use for reweighting
    pred_ls = []
    unique_elements, counts = np.unique(y, return_counts=True)
    total = sum(counts)
    class_weights = [1 - counts[i]/total for i in range(len(counts))]
    for data_i in range(X.shape[0]):
        # Leave only one out in test
        X_train, y_train = np.concatenate([X[:data_i], X[data_i+1:]]), np.concatenate([y[:data_i], y[data_i+1:]])
        X_test, y_test =  X[data_i].reshape(1, X.shape[1]), y[data_i].reshape(1, 1)
        # the classifier cond should be here
        if classifier == None:
            pred_point = special_knn_sep(class_weights, X_train, X_test, y_train, y_test)
        else:
            pred_point = other_clf_sep(classifier, X_train, X_test, y_train, y_test)
        pred_ls.append(pred_point)
    try:
        pred_ls = np.array(pred_ls)
    except:
        print("There is only one example of a pathogenicity class")
        raise ValueError
    return roc_auc_score(y, pred_ls)

# test this with the other clfs in query.py

# clf with full split precomputed
def classifier_full_split(X, y, classifier=None):
    # calculate class proportions and use for reweighting
    pred_ls = []
    unique_elements, counts = np.unique(y, return_counts=True)
    total = sum(counts)
    class_weights = [1 - counts[i]/total for i in range(len(counts))]
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    # now perform the classification task
    if classifier == None:
        predictions = special_knn_2d(class_weights, X_train, X_test, y_train, y_test)
    else:
        predictions = other_clf_sep(classifier, X_train, X_test, y_train, y_test)
    print(y_test.shape)
    print(predictions.shape)
    return roc_auc_score(y_test, predictions)

# now test this one

def special_knn(X, y, split=None):
    # calculate class proportions and use for reweighting
    pred_ls = []
    unique_elements, counts = np.unique(y, return_counts=True)
    total = sum(counts)
    print(counts, total, "counts and total")
    class_weights = [1 - counts[i]/total for i in range(len(counts))]
    # calc for each var in the X
    if split == None:
        for data_i in range(X.shape[0]):
            # need to concat
            train_X, train_y = np.concatenate([X[:data_i], X[data_i+1:]]), np.concatenate([y[:data_i], y[data_i+1:]])
            #print(y[data_i])
            test_X, test_y =  X[data_i].reshape(1, X.shape[1]), y[data_i].reshape(1, 1)
            if len(train_y) < 10:
                knn = NearestNeighbors(n_neighbors=len(train_y))
            else:
                knn = NearestNeighbors(n_neighbors=10)
            # make sure that the variant you want is excluded in train, and only that var in test
            # count the number of instances of each class, if anyone is 1, return "NaN"
            # Get unique elements and their counts
            unique_elements, counts = np.unique(train_y, return_counts=True)
            knn.fit(train_X, train_y) # not using anything held out
            # now get the nearest neighbors out, based on the test_X query
            neighbors = knn.kneighbors(test_X, return_distance=False)
            # should be np array, check for shape here
            neighbor_classes = np.take(train_y, neighbors, 0) # final arg is axis number, check to make sure which is which
            # broadcast the multiplication with your class_weights
            #print(unique_elements, "subset unique")
            #print(counts, "subset class count")
            #print(class_weights, "class weights")
            #print(len(neighbor_classes[neighbor_classes == 0]), "number benign in neighbors")
            class_votes = [class_weights[i] * len(neighbor_classes[neighbor_classes == unique_elements[i]]) for i in range(len(unique_elements))]
            #print(class_votes, unique_elements[np.argmax(class_votes)], "class votes")
            # make the prediction based on the reweighted votes 
            pred_ls.append(unique_elements[np.argmax(class_votes)])
    else:
        # must have train_X, train_y, test_X, test_y information
        train_X, train_y, test_X, test_y = split # tuple unrolled, so pass a tuple where each entry is np array
        unique_elements, counts = np.unique(train_y, return_counts=True)
        if len(train_y) < 10:
            knn = NearestNeighbors(n_neighbors=len(train_y))
        else:
            knn = NearestNeighbors(n_neighbors=10)
        # make sure that the variant you want is excluded in train, and only that var in test
        # count the number of instances of each class, if anyone is 1, return "NaN"
        # Get unique elements and their counts
        unique_elements, counts = np.unique(train_y, return_counts=True)
        knn.fit(train_X, train_y) # not using anything held out
        # now get the nearest neighbors out, based on the test_X query
        neighbors = knn.kneighbors(test_X, return_distance=False)
        #print(neighbors.shape, "neighbor shape")
        # should be np array, check for shape here
        neighbor_classes = np.take(train_y, neighbors, 0)
        #print(unique_elements, "subset unique")
        #print(counts, "subset class count")
        #print(class_weights, "class weights")
        # for arr in 2d arr
        for j in range(neighbor_classes.shape[0]):
        # if arr has more 0 than 1, call as 0 , o.w. 1
            class_votes =[class_weights[i]*len(neighbor_classes[j][neighbor_classes[j] == unique_elements[i]]) for i in range(len(unique_elements))]
            #print(class_votes, unique_elements[np.argmax(class_votes)], "class votes")
            # make the prediction based on the reweighted votes
            pred_ls.append(unique_elements[np.argmax(class_votes)])
        # do this for whole vect, returning a 2d arr.shape[0] vect
        return roc_auc_score(test_y, np.array(pred_ls))
    try:
        pred_ls = np.array(pred_ls)
    except:
        print("There is only one example of a pathogenicity class")
        raise ValueError
    print(y.shape)
    print(pred_ls.shape)
    return roc_auc_score(y, pred_ls)

# I want you to build a CI pipeline for this with Github Actions
# use the shell scripts you made

# move the knn_and roc code here
def knn_and_roc(X, y, multi_class='raise', average="macro", classifier=None, split=None): #KNeighborsClassifier(n_neighbors=5)): 
    # raise means it will complain about multiclass until overridden
    pred_ls = []
    # calc for each var in the X
    if split == None:
        for data_i in range(X.shape[0]):
            # Classify that variants with KNN (e.g. for K=5), i.e. assign each variant with a score 0<=s<=1 
            # which indicates the fraction of its K closest variants that are pathogenic (excluding the variant itself of course).
            if classifier != None:
                #print(classifier)
                knn = classifier_space[classifier]
            else:
                knn = KNeighborsClassifier(n_neighbors=5) # okay so now we need to reweight by  
            # need to concat
            train_X, train_y = np.concatenate([X[:data_i], X[data_i+1:]]), np.concatenate([y[:data_i], y[data_i+1:]])
            #print(y[data_i])
            test_X, test_y =  X[data_i].reshape(1, X.shape[1]), y[data_i].reshape(1, 1)
            #make sure that the variant you want is excluded in train, and only that var in test
            # count the number of instances of each class, if anyone is 1, return "NaN"
            # Get unique elements and their counts
            unique_elements, counts = np.unique(train_y, return_counts=True)
            if 1 in counts and clf_names[classifier] == "Logistic Regression":
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
    else:
        # must have train_X, train_y, test_X, test_y information
        train_X, train_y, test_X, test_y = split # tuple unrolled, so pass a tuple where each entry is np array
        if classifier != None:
            #print(classifier)
            knn = classifier_space[classifier]
        else:
            knn = KNeighborsClassifier(n_neighbors=5)
        unique_elements, counts = np.unique(train_y, return_counts=True)
        if 1 in counts and clf_names[classifier] == "Logistic Regression":
            return "NaN"
        knn.fit(train_X, train_y) # not using anything held out
        if multi_class=='ovr':
            pred_ls.append(knn.predict_proba(test_X))
        else:
            pred_ls.append(knn.predict(test_X))
        pred_ls = np.array(pred_ls)
        return roc_auc_score(test_y, pred_ls, multi_class=multi_class, average=average)
    try:
        pred_ls = np.array(pred_ls)
    except:
        print("There is only one example of a pathogenicity class")
        raise ValueError
    if multi_class=='ovr':
        pred_ls = pred_ls.reshape(pred_ls.shape[0], pred_ls.shape[-1])
    return roc_auc_score(y, pred_ls, multi_class=multi_class, average=average) 


# dump get_all_roc and move the partition logic elsewhere
def get_all_roc(X, y, path_name_in_df, roc_type, multi_class, average, classifier=None, split=None):     
    path_name_ls = ['Benign', 'Pathogenic', 'AR-GOF', 'AR-LOF', 'AD-LOF', 'AD-GOF', 'ADAR-LOF', 'ADAR-GOF']
    k = 0
    if split == None:
        if multi_class == "raise":
            print(roc_type +" ROC-AUC: " + str(knn_and_roc(X, y, classifier=classifier)))       
        else:
            # need to modify knn_and_roc behavior for multiclass
            # partition into clinvar part
            if 0 in path_name_in_df and 1 in path_name_in_df:
                clinvar_X = X[(y == 0) | (y == 1)]
                clinvar_y = y[(y == 0) | (y == 1)]
                if classifier == None:
                    print(1) # replace knn_and_roc with special_knn for now
                    print(roc_type  + " kNN Clinvar ROC-AUC: " +  str(special_knn(clinvar_X, clinvar_y)))
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
                print(3) # replace knn_and_roc with special_knn for now
                print(roc_type, "kNN LOF-GOF ROC-AUC:",  str(special_knn(lof_gof_X, lof_gof_y)))
            else:
                print(4)
                print(roc_type, clf_names[classifier], "LOF-GOF ROC-AUC:",  str(knn_and_roc(lof_gof_X, lof_gof_y, classifier=classifier)))
    # path_name_ls[i] +" is "+ str(roc_auc_ls[k]))
    else:
        print("split not None")
        print(multi_class, average, "mc and avg")
        # assume there are only binary splits
        #train_X, train_y, test_X, test_y = split
        return knn_and_roc(X, y, multi_class=multi_class, average=average, classifier=classifier, split=split)


# run G-NB, RF, LR; return labels and compute ROC_AUC later
global classifier_space 
classifier_space = [GaussianNB(), 
                    RandomForestClassifier(min_samples_split=5, n_jobs=-1), # was 2, 5
                    LogisticRegression(n_jobs=-1)]

#classifier_space = [None,
#                    GaussianNB(),
#                    RandomForestClassifier(min_samples_split=5, n_jobs=-1), # was 2, 5
#                    LogisticRegression(n_jobs=-1)]
global clf_names 
clf_names = [
        #"Class Reweighted kNN", 
        "Gaussian NB", "Random Forest", "Logistic Regression"]

def one_gene_clf(X, y, path_name_in_df, roc_type, multi_class,average):
    # one gene train/test -- use the knn_and_roc function
    # do wrt to both label sets -- needs to be passed in
    # I think we should just pass around the clf index, rather than the object itself
    for i in range(len(classifier_space)):
        # make sure the print statement says which classifier it is
        get_all_roc(X, y, path_name_in_df, roc_type, multi_class, average, classifier=i)
        
# now just need to ensure all embedds are read in and split is computed appropriately, then feed in with modified split arg
# all gene train/test
def all_gene_clf(X, y, path_name_in_df, gene_space,multi_class,average):
    full_X_train, full_X_test, full_y_train, full_y_test = [], [], [], []
    # for each classifier, .fit(X,y)
    for clf in classifier_space:
        get_all_roc(X, y, path_name_in_df, roc_type, multi_class, average, classifier=i)

# need to call special knn separate on all gene scale

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

