import warnings
warnings.filterwarnings('ignore')
import os, argparse, ast
from sklearn.neighbors import KNeighborsRegressor
import torch
import subprocess
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from scipy import stats
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
# you should add tqdm: two bars: one for assay out of total, one for sample out of total


def get_SM_PG(filter_str):
    all_human_files = glob.glob(filter_str) #'ProteinGym_substitutions/*')#HUMAN*') # I forgot these are only human genes
    filtered_files = [f for f in all_human_files if 'indel' not in f]
    ls_of_df = []
    for file in filtered_files:
        # read in file
        df = pd.read_csv(file)
        df["assay"] = file.split("/")[-1]
        df.mutant = df.mutant.unique()
        # bake in the gene name as a column
        df["gene"] = file.split("/")[-1].split("_")[0] + "_" + file.split("/")[-1].split("_")[1]
        try:
            #print(file.split("/")[1], len(df.index))
            ls_of_df.append(df[["gene", "mutant", "assay", "mutated_sequence", "DMS_score"]])
        except:
            print(file, "did not have mutant name, only sequence")
            print(df.columns)
            pass 
    # do this to get an ls of dfs 
    # then concatenate the dfs -- keep this for later to compare with clinvar mutations 
    all_gene_muts = pd.concat(ls_of_df)
    all_sm = all_gene_muts[~all_gene_muts['mutant'].str.contains(":")]
    # remove the 2 problem proteins
    all_sm = all_sm[~((all_sm.gene == "P53_HUMAN") | (all_sm.gene == "SPIKE_SARS2"))]
    # add back the WT sequence
    all_sm = add_WT_col(all_sm)
    # add saving at filepath
    return all_sm

def missense_to_WT(AA_str, edit):
    original_AA = edit[0]
    change_AA = edit[-1]
    location = int(edit[1:-1]) -1 # mutations are 1 indexed! -- in this file double check
    # size of prot seq changes between revisions
    if location > len(AA_str):
        return False
    elif AA_str[location] != change_AA: # the indexing is off by one?
        return False
    AA_str = AA_str[:location] + original_AA + AA_str[location+1:]
    #  print([(AA[i], i, editted_AA[i]) for i in range(len(editted_AA)) if editted_AA[i] != AA[i]])
    return AA_str

def add_WT_col(all_sm): # df must have mutated_sequence and mutant cols
    # now we apply this function as a lambda to our df
    all_sm['WT_sequence'] = all_sm.apply(lambda row: missense_to_WT(row['mutated_sequence'], row['mutant']), axis=1)
    return all_sm

def load_LLR_scores(LLR_csv, subset, all_sm):
    # now get LLR and add it to the big df
    # now match the LLRs to the WT sequences
    # perform the prediction for the DMS assay
    LLR = pd.read_csv(LLR_csv)
    LLR = LLR.rename(columns={'seq_id': 'seq_ID' }, inplace=False)
    # join on seqID with subset df
    all_WT_LLR = pd.merge(subset, LLR, on=['seq_ID']) 
    all_WT_LLR = all_WT_LLR.rename(columns={'mut_name': 'mutant' }, inplace=False)
    # now expand this to whole dataset based on gene, mut_name
    all_sm_LLR = pd.merge(all_sm, all_WT_LLR, on=['gene', 'mutant', 'WT_sequence'])
    return all_sm_LLR

def load_ESM_embeds(loaded_data, unique_mut_seqs, all_sm):
    esm_embeds = pd.DataFrame.from_dict(loaded_data, orient='index')
    esm_embeds["seq_ID"] = esm_embeds.index.astype('int64')
    # needs to be made flexible for whatever layer does the embedding
    # we remove this now, so we can accomodate several columns at once
    #esm_embeds = esm_embeds.rename(columns={33: 'esm_embed' }, inplace=False)
    # now merged based on index
    esm_embeds_with_genes = pd.merge(unique_mut_seqs, esm_embeds, on=['seq_ID'])
    # now we go from all unique esm embeds to all entries in the original df: all_sm
    all_sm_with_esm = pd.merge(all_sm, esm_embeds_with_genes, on=['gene', 'mutated_sequence']) # this is sufficient :)
    return all_sm_with_esm

def read_in_pt(filepath, embed_type, folder=False):
    # if full object not here:
    if folder:
        # given dir, read in as big dictionary -- needs glob
        all_files = glob.glob(f'{filepath}/*.pt')
        # Dictionary to hold data
        data_dict = {}
        for file in all_files:
            # Load the PyTorch tensor
            dic = torch.load(file)
            if embed_type == "mean":
                data_dict[dic['label']] = dic['mean_representations']
            else:
                # looks like {'label': '0_A24C', 'sliced_representations': {2: tensor([-0.7361, ..., 0.0608])}}
                data_dict[dic['label']] = dic['sliced_representations']
        # Save the dictionary as a .npy file
        np.save(f"{filepath}.npy", data_dict)
    # else read in the full object:
    else:
        data_dict = np.load(filepath, allow_pickle=True).item()
    return data_dict
    
def create_LLR_fasta(input_df, filepath, write=False):
    subset = input_df[['gene', 'WT_sequence']].drop_duplicates()
    subset["seq_ID"] = [ i for i in range(len(subset))]
    if write:
        with open(filepath, 'w') as f: # 'All_SM_PG.fasta'
            for index, row in subset.iterrows():
                f.write(f">{row['seq_ID']}\n")
                f.write(f"{row['WT_sequence']}\n")
    # write the file if needed, otherwise return the appropriate dataframe
    return subset

def create_ESM_fasta(input_df, filepath, write=False, short=True):
    # write the file if needed, otherwise return the appropriate dataframe
    if short:
        human_assays_only = input_df[input_df.gene.str.contains("HUMAN")]
        print(human_assays_only, 1)
        unique_human_muts = (human_assays_only[["gene", "mutated_sequence"]].drop_duplicates()) # 187997 seqs. easy
        print(unique_human_muts, 2)
        # so take this indices and 
        unique_mut_seqs = human_assays_only.loc[human_assays_only.index.intersection(
            unique_human_muts.index)][["gene", "mutant", "mutated_sequence"]]
        #unique_mut_seqs = human_assays_only.loc[unique_human_muts.index] #[["gene", "mutant", "mutated_sequence"]]
        unique_mut_seqs["seq_ID"] = [ i for i in range(len(unique_mut_seqs))]
        # this will eliminate long sequences -- only the slice ones will have this nomenclature.
        unique_mut_seqs = unique_mut_seqs[unique_mut_seqs.mutated_sequence.str.len() <= 1022]
        print(unique_human_muts, 4)
        #raise Error
    else: 
        unique_mut_seqs = input_df[['gene', 'mutated_sequence']].drop_duplicates()
        unique_mut_seqs["seq_ID"] = [ i for i in range(len(unique_mut_seqs))]
    if write:
        with open(filepath, 'w') as f: # 'All_SM_PG_esm.fasta'
            for index, row in unique_mut_seqs.iterrows():
                f.write(f">{row['seq_ID']}\n")
                f.write(f"{row['mutated_sequence']}\n")
    return unique_mut_seqs

def run_sh_command(command):
    # Run the command and capture the output
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # Decode bytes to string and print the output
    if process.returncode == 0:
        print("Success:")
        print(stdout.decode())
    else:
        print("Error:")
        print(stderr.decode())
# make sure this part works tomorrow.
# run the next slice layer after gym
# use this script to make graphs on AWS, don't move the files down to local.
# with another rounds of forward pass tomorrow, that should be it. If you start the forward passes early, this whole thing should be done by late Friday
# don't forget you have to make a new resume tomorrow and send it to UK biobank people. 
def assemble_full_df(filter_str, ESM_fasta_name, LLR_fasta_name, ESM_dir_name,
                     ESM_run=True, LLR_run=True, folder=False, 
                     esm_model="esm1b_t33_650M_UR50S", embed_type="mean", repr_layers=33):
    all_sm = get_SM_PG(filter_str)
    print(all_sm)
    unique_mut_seqs = create_ESM_fasta(all_sm, ESM_fasta_name,)# not ESM_run)
    print(unique_mut_seqs)
    # there must be a conditional to know if esm has already been run, so it knows the .pt exist
    #some(ESM_fasta_name, repr_layers, embed_type)Human_SM_PG_slice
#     if not ESM_run:
#         cmd =f"python3 extract.py {esm_model} {ESM_fasta_name} {ESM_dir_name} --repr_layers {repr_layers} --include {embed_type}"
#         run_sh_command(cmd)
    data_dict = read_in_pt(ESM_dir_name, embed_type, folder=folder)
    #print(data_dict)
    #print(data_dict)
    subset = create_LLR_fasta(all_sm, LLR_fasta_name,)# not LLR_run) # if LLR_run is false, write the fasta
    print(subset)
    # there must be a conditional to know if the LLR script has been run -- otherwise LLR_csv does not exist
    # make the LLR name
    stub = LLR_fasta_name.split(".")[0]
    LLR_csv = f"{stub}_LLR.csv"
    print(LLR_csv, LLR_run) # I think we were wondering if a suitable csv exists
    
    if not LLR_run:
        cmd = f"python3 esm-variants/esm_score_missense_mutations.py --input-fasta-file {LLR_fasta_name} --output-csv-file {LLR_csv}"
        run_sh_command(cmd)
    all_sm_LLR = load_LLR_scores(LLR_csv, subset, all_sm)
    print(all_sm_LLR, len(all_sm_LLR[all_sm_LLR.assay == "NKX31_HUMAN_Rocklin_2023_2L9R.csv"].index))
    print(5)
    raise Error
    all_sm_with_LLR_and_ESM = load_ESM_embeds(data_dict, unique_mut_seqs, all_sm_LLR)
    # now subset on Human only?
    all_sm_with_LLR_and_ESM = all_sm_with_LLR_and_ESM[all_sm_with_LLR_and_ESM.gene.str.contains("HUMAN")]
    # must happen after adding the ESM embeddings due to the nature of the fasta that generated this. Very dumb.
    print(all_sm_with_LLR_and_ESM, len(all_sm_with_LLR_and_ESM[all_sm_with_LLR_and_ESM.assay == "NKX31_HUMAN_Rocklin_2023_2L9R.csv"].index))
    print(6)
    # DO NOT save this. NEVER save a df with tensors in it.
    return all_sm_with_LLR_and_ESM

# stuff to perform chloe hsu augment
aa_to_int = {
    'M':1,
    'R':2,
    'H':3,
    'K':4,
    'D':5,
    'E':6,
    'S':7,
    'T':8,
    'N':9,
    'Q':10, 'C':11,
    'U':12,
    'G':13,
    'P':14,
    'A':15,
    'V':16,
    'I':17,
    'F':18,
    'Y':19,
    'W':20,
    'L':21,
    'O':22, #Pyrrolysine
    'X':23, # Unknown
    'Z':23, # Glutamic acid or GLutamine
    'B':23, # Asparagine or aspartic acid
    'J':23, # Leucine or isoleucine
    'start':24,
    'stop':25,
    '-':26,
}

def aa_seq_to_int(s):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    return [24] + [aa_to_int[a] for a in s] + [25]

def format_seq(seq,stop=False):
    """
    Takes an amino acid sequence, returns a list of integers in the codex of the babbler.
    Here, the default is to strip the stop symbol (stop=False) which would have 
    otherwise been added to the end of the sequence. If you are trying to generate
    a rep, do not include the stop. It is probably best to ignore the stop if you are
    co-tuning the babbler and a top model as well.
    """
    if stop:
        int_seq = aa_seq_to_int(seq.strip())
    else:
        int_seq = aa_seq_to_int(seq.strip())[:-1]
    return int_seq

# take a list of seqs and converts to a batch of seqs
def format_batch_seqs(seqs):
    maxlen = -1
    for s in seqs:
        if len(s) > maxlen:
            maxlen = len(s)
    formatted = []
    for seq in seqs:
        pad_len = maxlen - len(seq)
        padded = np.pad(format_seq(seq), (0, pad_len), 'constant', constant_values=0)
        formatted.append(padded)
    return np.stack(formatted)
# converts ls of seqs -> batch -> one hot encodes seq batch
def seqs_to_onehot(seqs):
    seqs = format_batch_seqs(seqs)
    X = np.zeros((seqs.shape[0], seqs.shape[1]*24), dtype=int)
    for i in range(seqs.shape[1]):
        for j in range(24):
            X[:, i*24+j] = (seqs[:, i] == j)
    return X

# now the code necessary to perform prediction
def get_splits(input_df, SEED, splits):
    # Set a static test set of 20% of the entire dataset
    train_overall, test = train_test_split(input_df, test_size=0.2, random_state=SEED)
    # Set the various sizes for the training splits
    # splits = [0.1, 0.3, 0.5, 0.8, 10, 25, 50, 100, 250, 500]
    # Generate training sets of various sizes
    train_splits = {}
    for split in splits:
        # If split == 0, the training set is an empty dataframe
        if split == 0:
            train_splits[split] = pd.DataFrame(columns=df.columns)
        # sometimes split too big given N of assay
        elif split > 1 and split <= len(train_overall):
            train_split = train_overall.sample(n=split, random_state=SEED)
            train_splits[split] = train_split
        else:
            # split would take 80% of the overrall train, not 100% of the 80%
            _, train_split = train_test_split(train_overall, test_size=split*len(input_df.index), random_state=SEED)
            train_splits[split] = train_split
    return train_splits, test

def get_X(new_subset, estimator_list): # ls of str
    component_dict = dict()
    for estimator in estimator_list:
        # gather necessary components in dict
        if "ESM" in estimator: # this will need editing in the future
            ESM_subset = (np.vstack(new_subset.esm_embed.values)) # (168, 1280)
            component_dict["ESM"] = ESM_subset
        if "LLR" in estimator:
            LLR_subset = (new_subset.esm_score.values)
            LLR_subset = LLR_subset.reshape(LLR_subset.shape[0], 1)
            component_dict["LLR"] = LLR_subset
        if "one-hot" in estimator:
            # add the augment to LLR -- just a one hot encoding
            seqs = new_subset.mutated_sequence.values
            one_hot = seqs_to_onehot(seqs) # seems to work fine
            component_dict["one-hot"] = one_hot
    # now that all components are gathered, create the X_arr combinations based on the estimator list
    X_arr = []
    for estimator in estimator_list:
        # for simple ones
        # needs to be modified to acount for knn_ESM and LLR_direct
        # break the str into pieces based on some separator, let's say "_"
        pieces = estimator.split("_")
        combo = all(piece in component_dict for piece in pieces)
        #print(estimator, combo)
        #if estimator in component_dict.keys():
        if not combo:
            bools = [piece in component_dict for piece in pieces]
            estimator_indices = [i for i, val in enumerate(bools) if val]
            if len(estimator_indices) > 1:
                raise NotImplementedError
            X_arr.append(component_dict[pieces[estimator_indices[0]]])
        else: # otherwise they are combinations
            all_together = []
            for piece in pieces:
                all_together.append(component_dict[piece])
            # concatenate and place in X_arr
            X_arr.append(np.concatenate(all_together, axis=1))
    # y_arr is just DMS_score
    y_arr = new_subset.DMS_score.values
    y_arr = y_arr.reshape(y_arr.shape[0], 1)
    return X_arr, y_arr

def train_and_predict(X_train, y_train, X_test, y_test, corre_ls, lm):
    # Train the model -- default alpha is 1.0 as desired.
    lm.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = lm.predict(X_test)
    # use spearman's corre to get performance
    res = stats.spearmanr(y_pred, y_test) #[:,j], y_test[:,j])
    corre_ls.append(res.correlation)  
    return corre_ls

# add  regressor_list [len(estimator_list)] where element is in the set {LLR_direct, knn, Ridge}
def training_loop(human_assays_only, splits, seed_number, threshold, estimator_list): 
    seed_list = [i for i in range(seed_number)]
    unique_assays = human_assays_only['assay'].unique()
    subset = []
    # filter the assay list by index len and threshold
    for assay in unique_assays:
        if len(human_assays_only[human_assays_only.assay == assay].index) > threshold:
            #print(assay, len(human_assays_only[human_assays_only.assay == assay].index))
            subset.append(assay)
    unique_assays = subset
    # need to score across assays
    for_graphs = {key: [[] for i in range(len(unique_assays))] for key in splits}
    # keep track of assay order
    categories = []
    # we still need to average over seeds but not over splits
    # use train split as basis for all stuff
    # tqdm here -- this is straightforward
    # Setup the outer tqdm bar
#     total_sum = sum(lst)
#     cumulative_sum = 0
#     with tqdm(total=len(lst), desc="List Progress", position=0) as pbar1:
#         with tqdm(total=total_sum, desc="Cumulative Sum", position=1, leave=False) as pbar2:
#             for item in lst:
#                 cumulative_sum += item
#                 pbar2.update(item)
#                 pbar1.update(1)
    for as_i in range(len(unique_assays)):
        assay = unique_assays[as_i]
        new_subset = human_assays_only[human_assays_only['assay'] == assay]
        print(assay, len(new_subset.index))
        # make a dictionary of split:[]
        split_store =  {key: [] for key in splits}
        for seed in seed_list:
            # get splits
            train_splits, test = get_splits(new_subset, seed, splits)
            for fraction, train_split in train_splits.items():
                # get the X arr for train
                X_train, y_train = get_X(train_split, estimator_list)
                # same for test
                X_test, y_test = get_X(test, estimator_list)
                # now define all_spearmans to catch
                all_spearmans = [[] for k in range(len(X_train))]
                for i in range(len(all_spearmans)):
                    if estimator_list[i] == "LLR_direct": 
                        res = stats.spearmanr( X_test[i], y_test)
                        # is a single val acceptable here? Yes
                        all_spearmans[i].append(res.correlation)
                    else:
                        if "knn" in estimator_list[i]:
                            regressor = KNeighborsRegressor()
                        # ridge, LLR_direct, knn regressor, then start probing combinations
                        else:
                            regressor = Ridge() # estimator_list[i]
                        all_spearmans[i] = train_and_predict(X_train[i], y_train, X_test[i], y_test, 
                                                                 all_spearmans[i], regressor)
                split_store[fraction].append(all_spearmans)
        # 20, then 7
        for key, value in split_store.items():
            split_store[key] = np.array(value)
        #print(split_store[0.1].shape) # 20,7,1
        # after collecting all splits
        # catch the assay in categories
        categories += [assay]
        # so make a different graph for each split?
        for k in split_store.keys():
            for i in range(split_store[k].shape[1]):
                arr = np.mean(split_store[k][:, i]) # should be shape of 20 for the input
                if np.all(np.isnan(arr)):
                    arr = np.zeros_like(arr)
                for_graphs[k][as_i].append(arr)
        # convert to np arrays, account for N < 500 for an assay
    #print(for_graphs) # fraction: [assay[estimator]]
    # these definitely should get saved somehow. Generating this is a pain.
    return categories, for_graphs


def write_out_pred_results(data, assay_list, estimator_list, outname):
    # Convert the nested dictionary into a list of records
    records = []
    for split, assays in data.items():
        for assay_idx in range(len(assay_list)):
            for est_idx in range(len(estimator_list)):
                record = {
                    "split": split,
                    "assay": assay_list[assay_idx], 
                    "estimator": estimator_list[est_idx], 
                    "correlation_score": assays[assay_idx][est_idx]
                }
                records.append(record)
    # Convert the list of records into a DataFrame
    df = pd.DataFrame(records)
    # write it out
    df.to_csv(outname, index=False)
# now the graph making code --
# want the final graph to be all splits with all estimators averaged over all seeds
# one graph for each ESM repr type. Will be 8 graphs in total


# okay let's modify this to make one graph across the whole assay set
def make_graphs(estimator_list, df_path, layer_num, embed_type):
    # need to define the fig and ax here
    df = pd.read_csv(df_path)
    # Using the 'tab10' colormap for high contrast
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    # Create a separate figure for the scatterplot
    scatter_fig, scatter_ax = plt.subplots(figsize=(10, 6))
    # add alpha value
    alpha_value = 0.3
    # splits will be X axis
    splits = df.split.unique()
    # I think flipping the loops should resolve the problem
    for j in range(len(estimator_list)):
        sub_sub = df[df.estimator == estimator_list[j]][["split", "correlation_score"]].values #df[df.split == splits[i]]
        X, y = sub_sub[:, 0], sub_sub[:, 1]
#         scatter_ax.scatter(X, y, 
#                         s=100, color=colors[j], label=estimator_list[j])
        sns.lineplot(x=X, 
                 y=y, 
                 color=colors[j], ax=scatter_ax, label=estimator_list[j])
#         raise Error
#         for i in range(len(splits)):
#             y = sub_sub[sub_sub.split == splits[i]].correlation_score.values
#             scatter_ax.scatter([splits[i]], np.mean(y), 
#                         s=100, color=colors[j], label=estimator_list[j])
            # plots line of best fit between splits for each estimator
#             sns.lineplot(x=[splits[i]]*len(y), 
#                  y=y, 
#                  color=colors[j], ax=scatter_ax, ci='sd')
            
    
#     # each assay will have its own graph
#     assays = df.assay.unique()
#     for assay in assays:
#         subset = df[df.assay == assay]
#         # Using the 'tab10' colormap for high contrast
#         colors = plt.cm.tab10(np.linspace(0, 1, 10))
#         # Create a separate figure for the scatterplot
#         scatter_fig, scatter_ax = plt.subplots(figsize=(10, 6))
#         # add alpha value
#         alpha_value = 0.3
#         # for all of a split
#         for i in range(len(splits)):
#             sub_sub = subset[subset.split == splits[i]]
#             for j in range(len(estimator_list)):
#                 #print(splits[i], sub_sub[sub_sub.estimator == estimator_list[j]].correlation_score.values)
#                 scatter_ax.scatter([splits[i]], sub_sub[sub_sub.estimator == estimator_list[j]].correlation_score.values, 
#                             s=100, color=colors[j], label=estimator_list[j])
#                 # plots line of best fit between splits for each estimator
#                 sns.lineplot(x=splits, 
#                      y=subset[subset.estimator == estimator_list[j]].correlation_score.values, 
#                      color=colors[j], ax=scatter_ax, ci='sd')
                # add uncertainty
#                 scatter_ax.fill_between([splits[i]], sub_sub[sub_sub.estimator == estimator_list[j]].correlation_score.values - 0.1,
#                                         sub_sub[sub_sub.estimator == estimator_list[j]].correlation_score.values + 0.1, 
#                                         color=colors[j], alpha=alpha_value)
#                 scatter_ax.plot(splits, 
#                                  subset[subset.estimator == estimator_list[j]].correlation_score.values, 
#                                  color=colors[j])
    scatter_ax.set_xticklabels(scatter_ax.get_xticklabels(),rotation=45, ha='right')
    scatter_ax.set_xlabel('Number of Training Points')
    scatter_ax.set_ylabel("Spearman's Correlation with DMS scores")
    scatter_ax.set_title(f"Spearman's Correlation for SM Human Assays, Layer: {layer_num}, Type: {embed_type}")#; N={size}") try to get N next time
    # Get existing handles and labels
    handles, labels = scatter_ax.get_legend_handles_labels()
    # Remove duplicates
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    unique_handles, unique_labels = zip(*unique)
    # Set the legend with unique handles and labels
    scatter_ax.legend(unique_handles, unique_labels, loc='upper left', bbox_to_anchor=(1, 1))
    scatter_fig.tight_layout()
    # Save or show the scatterplot figure
    plt.savefig(f"SM Human Assays_{layer_num}_{embed_type}.png", facecolor='white')
    plt.close(scatter_fig)
    raise Error

# extra logic: train? bool, 
# ESM_run=True, LLR_run=True, folder=False

# esm_model, embed_type="mean", repr_layers=33
# location of assay files, esm fasta location, WT fasta location
# pred results file path
# 11 args sheesh

# now we need to add things that check this
def create_parser():
    parser = argparse.ArgumentParser(
        description="Trains regressor to predict DMS score from Protein Gym Variants"  # noqa
    )
    parser.add_argument( # assert it is in the human gene set based on the PROTEOME_FASTA_FILE_PATH
        "--ESM_done",
        #type=bool, # I believe there was an issue getting bools args to work
        #default=True,
        action="store_true", # these are for bool args, use this instead of default -- let's do this 
        # and  commit to ESM variant sweep
        help="Boolean that says if ESM embeddings have already been extracted. Default True.",
    )
    parser.add_argument( # ["esm_only", "esm_vae", "all", "None"]
        "--LLR_done",
        #type=bool,
        action="store_true",
        help="Boolean that says if LLRs have already been extracted. Default True.",
    )
    parser.add_argument( # ["esm_only", "esm_vae", "all", "None"]
        "--already_trained",
        #type=bool,
        action="store_true",
        help="Boolean that says if predictions have already been computed. Default True.",
    )
    parser.add_argument( # ["esm_only", "esm_vae", "all", "None"]
        "--embed_loc",
        type=str,
        default="None",
        help="Tells the pipeline where the ESM embeddings are or where to write them. By Default assumes None",
    )
    parser.add_argument( # ["esm_only", "esm_vae", "all", "None"]
        "--folder",
        #type=bool,
        action="store_true",
        help="Tells if embed_loc is a folder or not. Default True.",
    )    
    parser.add_argument( 
        "--esm_model",
        type=str, 
        default="esm1b_t33_650M_UR50S", 
        choices=["esm1b_t33_650M_UR50S"], # change this to ESM2 eventually
        help="Tells the pipeline which ESM 1B model to use. By Default uses 650 Million param model",
    )     
    parser.add_argument( # number of random mutations to augment all point missense clinvar mutations 
        "--embed_type",
        type=str, 
        default="mean",# make an edge case to accomodate the full mutation space, make "full" 
        choices=["mean", "slice",], # "contacts"],
        help="Tells the pipeline what kind of ESM embedding. By default is mean.",
    )
    parser.add_argument( # number of random mutations to augment all point missense clinvar mutations 
        "--layer_num",
        type=int, 
        default=33,# make an edge case to accomodate the full mutation space, make "full" 
        choices=[2, 9, 21, 33],
        help="Tells the pipeline which layer the ESM embedding is extracted from. By default is 33.",
    )
    parser.add_argument( # number of random mutations to augment all point missense clinvar mutations 
        "--assay_dir",
        type=str, 
        default="ProteinGym_substitutions/*",# make an edge case to accomodate the full mutation space, make "full" 
        help="Tells the pipeline where the Protein Gym Assays are. Default ProteinGym_substitutions/*.",
    )
    parser.add_argument( # number of random mutations to augment all point missense clinvar mutations 
        "--esm_fasta",
        type=str, 
        default="None",# make an edge case to accomodate the full mutation space, make "full" 
        help="Tells the pipeline the location of the Fasta to generation ESM embeddings. Default None.",
    )
    parser.add_argument( # ["llr_only, "llr_pos", "clinvar_only", "all"]
        "--WT_fasta",
        type=str,
        default="None",
        help="Tells the pipeline the location of the Wild Type Fasta to compute LLR. Default None.",
    ) # pred results file path
    parser.add_argument( # ["llr_only, "llr_pos", "clinvar_only", "all"]
        "--pred_results",
        type=str,
        default="None",
        help="Tells the pipeline where to store or read the prediction results. Default None",
    )
    
    return parser

# then make the graph code more modular
def main():  
    parser = create_parser()
    args = parser.parse_args()
    estimator_list = [
            "one-hot", "LLR_one-hot",
            "ESM", "LLR_ESM",
            "ESM_one-hot", "ESM_one-hot_LLR", "LLR_direct", "knn_ESM"]
    if not args.already_trained:
        # first get the Human assay SM dataframe.
        # and change the ESM dir names to match the naming convention
        human_assays_only = assemble_full_df(#"ProteinGym_substitutions/*", "All_SM_PG_esm.fasta", "All_SM_PG.fasta", 
                                             args.assay_dir, args.esm_fasta, args.WT_fasta,
                                             #"All_PG_SM_embed.npy"
                                             args.embed_loc, ESM_run=args.ESM_done, LLR_run=args.LLR_done, 
                                             folder=args.folder, # "esm1b_t33_650M_UR50S","mean", 33
                                             esm_model=args.esm_model, embed_type=args.embed_type, repr_layers=args.layer_num)
        # rename the esm embeds 
        human_assays_only = human_assays_only.rename(columns={args.layer_num: 'esm_embed' }, inplace=False)
        # now we perform the splits and train the estimators
        #splits = [0.1, 0.3, 0.5, 0.8, 
        splits = [10, 25, 50, 100, 250, 500]
        seed_number = 20
        threshold = 625# 313 # this allows 500 train points while still having 20% to validate on
        # okay still need LLR direct and some knn ones in here
        # estimator list should be specified here
        categories, for_graphs = training_loop(human_assays_only, splits, seed_number, threshold, estimator_list)
#         print(categories)
#         print(for_graphs)
#         print(estimator_list)
        # write out categories and for_graphs 
        # categories is just ls of assays
        # for_graphs is dict of ls of ls
        #  {fraction: [assay[estimator]]}
        # let's save this is as a df
        # also write out N for each assay
        write_out_pred_results(for_graphs, categories, estimator_list, args.pred_results)# "Large_Human_Results.csv")
#     N_list = []
#     for assay in categories: # you will want this to be written out as well so you don't have to recompute as much
#         N_list.append(len(human_assays_only[human_assays_only.assay == assay].index))
    # then we graph the output of the estimators
    # need layer num and embed type in the title
    make_graphs(estimator_list, args.pred_results, args.layer_num, args.embed_type)
    # contingent upon this working, we simply deploy to each suitable .npy
    # it will take time for the npys to be made.

if __name__ == "__main__":
    main()