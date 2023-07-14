import os, argparse, ast

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

import torch.utils.data as data_utils
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from all_genes import *
from read_embeddings import *
from esm_VAE import *
from UMAP_graphs import *

def write_fasta(output_file, clinvar_muts):
    #"additional_PKD1.fasta"#"all_PKD1.fasta" #'all_mutations.fasta'
    # Open the output file in write mode
    with open(output_file, 'w') as file:
        # Iterate over the DataFrame rows
        for mut_name, AA in clinvar_muts: # was full
            # Write the sequence header line in FASTA format
            file.write(f'>{mut_name}\n')
            # Write the sequence
            file.write(f'{AA}\n')

def knn_and_roc(X, y, multi_class='raise', average="macro"): # raise means it will complain about multiclass until overridden
    pred_ls = []
    # calc for each var in the X
    for data_i in range(X.shape[0]):
        # Classify that variants with KNN (e.g. for K=5), i.e. assign each variant with a score 0<=s<=1 
        # which indicates the fraction of its K closest variants that are pathogenic (excluding the variant itself of course).
        knn = KNeighborsClassifier(n_neighbors=5) 
        # need to concat
        train_X, train_y = np.concatenate([X[:data_i], X[data_i+1:]]), np.concatenate([y[:data_i], y[data_i+1:]])
        #print(y[data_i])
        test_X, test_y =  X[data_i].reshape(1, X.shape[1]), y[data_i].reshape(1, 1)
        # make sure that the variant you want is excluded in train, and only that var in test
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
                
# an arg parser is a good idea
# python query.py P53 num_random=full
def create_parser():
    parser = argparse.ArgumentParser(
        description="Runs ESM, VAE, UMAP pipeline for chosen gene"  # noqa
    )
    parser.add_argument( # assert it is in the human gene set based on the PROTEOME_FASTA_FILE_PATH
        "--gene",
        type=str,
        help="Gene of interest with which to perform the analysis",
    )
    parser.add_argument( # ["esm_only", "esm_vae", "all", "None"]
        "--existing_embeddings",
        type=str,
        default="None",
        help="Tells the pipeline which embeddings already exist. By Default assumes None",
    )
    parser.add_argument( # ["esm_only", "esm_vae", "all", "None"]
        "--extra_labels",
        type=str,
        default="None",
        help="Tells the pipeline where additional labels beyond clinvar live. By Default assumes None",
    )    
    parser.add_argument( 
        "--df_preloaded",
        type=str, # True or False
        default="False", 
        help="Tells the model if the clinvar df is preprocessed embeddings. By Default assumes False",
    )     
    parser.add_argument( # number of random mutations to augment all point missense clinvar mutations 
        "--num_random",
        #type=int, # it will assume this is a str
        default=0, # make an edge case to accomodate the full mutation space, make "full" 
        help="Tells the model how many point missense mutations to add to the gene beyond clinvar mutations. By default is 0.",
    )
    parser.add_argument( # ["llr_only, "llr_pos", "clinvar_only", "all"]
        "--get_graphs",
        type=str,
        default="all",
        help="Which graphs to write out: the LLRs from the UMAP, the LLRs and pos of mutation on the UMAP, clinvar muations on the UMAP",
    )
    parser.add_argument( # ["llr_only, "llr_pos", "clinvar_only", "all"]
        "--esm_embed_dir",
        type=str,
        default="None",
        help="Tells the pipeline which directory to take the esm embeddings from. Assumes None by default.",
    )
    return parser


# not every gene name from clinvar can map to uniprot
def check_args(args):
    # read in uniprot list 
    shared_uniprot = open('final_uniprot_list.txt','r')
    uniprot_gene_names = []
    while True:
        # Get next line from file
        line = shared_uniprot.readline().strip()
        # if line is empty end of file is reached
        if not line:
            break
        uniprot_gene_names.append(line)
    assert args.gene in uniprot_gene_names, "Not in list of human genes from Uniprot Proteome" 
    #assert args.gene in clinvar_gene_names, "Not in list of human genes from Clinvar" 
    #assert args.df_preloaded 
    assert args.existing_embeddings in ["esm_only", "all", "None"], \
    "Existing Embedding Argument not in answer set. Acceptable answers are esm_only, all, or None." 
#     assert args.get_embed in ["esm_only", "all_but_esm", "all", "None"],  \
#     "Get Embedding Argument not in answer set. Acceptable answers are esm_only, all_but_esm, or all."

#     assert args.esm_embed_dir == "None" and args.existing_embeddings not in ["esm_only", "all"], "If esm embedding directory not specified, then esm embeddings do not exist. Likewise, if esm embedding directory specified then esm embeddings must exist."
    assert args.get_graphs in ["llr_only", "llr_pos", "clinvar_only", "all"],  \
    "Get Graphs Argument not in answer set. Acceptable answers are llr_only, llr_pos, clinvar_only, or all." 
    try: 
        rand_type = type(int(args.num_random)) 
    except:
        assert args.num_random == "full", \
    "Number of Random Mutations argument not in answer set. Acceptable answers are any integer or just full" 
    
# make a command line interface that will run the pipeline for any human gene
def main():
    parser = create_parser()
    args = parser.parse_args()
    # add a config file that must be defined
    # hard coding filepaths based on a common dir seems to be best way to avoid too many args
    # just make sure you make a dir with all such files
    PROTEOME_FASTA_FILE_PATH = "data/uniprot/human_reviewed.fa.gz" # keep on mounted volume
    ALL_CLINVAR = "all_missense.txt" # keep on mounted volume
    FASTA_FILE = args.gene + ".fasta"# should be derived from gene name
    MODEL_LOCATION = "esm1b_t33_650M_UR50S" # name of pretrained model, will download if not in torch cache
    # could also use a big boi model later, make sure it has the correct weights in the container
    VARIANT_SCORES_DIR = "ALL_hum_isoforms_ESM1b_LLR" #"data/esm1b_variant_scores/ALL_hum_isoforms_ESM1b_LLR" # keep on mounted volume
    # ESM INPUT PARAMS
    repr_layers = 33 # default arg in all that I've seen (final layer representation)
    toks_per_batch = 128 # could be increased all the way up to 4096, depends on RAM
    ESM_EMBEDDINGS = args.gene + "_esm_embed" if args.esm_embed_dir == "None" else args.esm_embed_dir
    #"P53_embeds"  #"new_new/new_new" # # args.gene +"esm_dir"# dir where the esm embeds will be dumped
    # loactions for other embeddings
    DELTA_EMBEDDINGS = args.gene +"_delta_embeddings.npy"
    VAE_EMBEDDINGS = args.gene +"_vae_embeddings.npy"
    UMAP_EMBEDDINGS = args.gene +"_umap_embeddings.npy"# graphs will output into current working directory 
    DF_LOCATION = "clinvar_preprocessed.csv"
    RECT_DF_LOCATION = args.gene + "_final_df.csv"
    C2P = "clinvar_2_uniprot.tsv" # keep on mounted volume
    # more logic to prevent excess waiting
    # all of these function calls are from all_genes.py
    uniprot_records, gene_name_to_uniprot_records = get_uniprot(PROTEOME_FASTA_FILE_PATH)
    # check if df preloaded
    if args.df_preloaded == "True":
        # read in the df
        df = pd.read_csv(DF_LOCATION)
    else:
        df = get_clinvar_df(ALL_CLINVAR)
        # save it for next time
        df.to_csv(DF_LOCATION)
    print("Clinvar DataFrame Loaded")
    # df is Unnamed: 0    Name mutation_name gene_name  clinvar_label GRCh38Chromosome GRCh38Location   Name_edit
    df = df[["mutation_name", "gene_name", "clinvar_label"]]
    # create a path_name col and use this for the graphs
    df['path_name'] = np.where(df['clinvar_label'] == 0, 'Benign', 'Pathogenic')
    # if additional labels
    if args.extra_labels != "None": # now this needs to get worked on
        extra_df = pd.read_csv(args.extra_labels)
        # Gene	Uniprot_variant	Inheritance	Disease_mechanism
        # make path name from all combos of Inheritance	Disease_mechanism
        extra_df["path_name"] = extra_df["Inheritance"] + "-" + extra_df["Disease_mechanism"]
        #"mutation_name", "gene_name", "clinvar_label"
        extra_df = extra_df.rename(columns={'Uniprot_variant': 'mutation_name', 'Gene': 'gene_name'})
        add_labels = extra_df["path_name"].unique() # ['AR-GOF' 'AR-LOF' 'AD-LOF' 'AD-GOF' 'ADAR-LOF' 'ADAR-GOF']
        df_ls = []
        for i in range(len(add_labels)):
            subset_df = extra_df[extra_df["path_name"] == add_labels[i]] 
            subset_df["clinvar_label"] = i+2
            df_ls.append(subset_df)
        extra_df = pd.concat(df_ls)
        extra_df = extra_df[["mutation_name", "gene_name", "path_name", "clinvar_label"]]
        # need to run some assertions that the dfs will be compatible to concat -- do ASAP
        # concat the extra df to the main df
        df = pd.concat([df, extra_df])    
    print("Extra Labels Loaded")
    # verify inputs with check_args
    check_args(args) #, uniprot_gene_names, clinvar_gene_names)
    # this must subset on the uniprot name of the variant in clinvar
    clinvar2uniprot = pd.read_csv(C2P, sep="\t")
    # convert to dict
    dictionary = dict(zip(clinvar2uniprot['Entry'], clinvar2uniprot['From']))
    # subset the df on the gene
    df = df[df["gene_name"] == dictionary[args.gene]]
    #print(df.clinvar_label.unique())
    #df = df[df["gene_name"] == args.gene]
    # check how many pathogen label types there will be, activate the multiclass flag if > 2
    # compute the len of protein, must be only one gene
    gene_name = uniprot_records[uniprot_records.gene.isin([args.gene])].gene.unique()[0]
    # gene_name
    (_, gene_uniprot_record), = gene_name_to_uniprot_records.get_group(gene_name).iterrows()
    aa_length = len(gene_uniprot_record['seq'])
    # convert gene to gene_list
    if args.existing_embeddings in [ "esm_only", "None"]:
        if args.existing_embeddings == "None":
            genes = get_init_tuples([args.gene], uniprot_records, gene_name_to_uniprot_records)
            # we assume the ls will always be size one as we only do one gene at time now
            genes = [(dictionary[genes[0][0]], genes[0][1])]        
            clinvar_muts = get_var_cent_tuples(genes, df)
            # so it should have the vars by here
            # random size is another input
            clinvar_muts = get_ESM_tuples(genes, clinvar_muts, args.num_random)
            # need fasta_file name as input
            write_fasta(FASTA_FILE, clinvar_muts)
            print("Fasta written")
            # can run directly from command line as long as the extract.py script is in same dir -- do this instead
            # the default script has a gpu autodetect
            print("python3 extract.py "+MODEL_LOCATION + " " + FASTA_FILE + " " + ESM_EMBEDDINGS +
                      " --toks_per_batch "+ str(toks_per_batch) + " --repr_layers " + str(repr_layers) + " --include per_tok")
            os.system("python3 extract.py "+MODEL_LOCATION + " " + FASTA_FILE + " " + ESM_EMBEDDINGS +
                      " --toks_per_batch "+ str(toks_per_batch) + " --repr_layers " + str(repr_layers) + " --include per_tok")
        # these are from read_embeddings.py
        # read in esm embeddings, needs dir path and df to append file paths to
        print(df.clinvar_label.unique())
        #print(len(df.mutation_name.unique())) # 444, but did these get included on the esm pass?
        reprs = read_esm_reprs(ESM_EMBEDDINGS, df)
        #print(reprs[reprs.clinvar_label.notnull()].clinvar_label) # 1059 -- it goes down to like 335 now
        print("ESM embeddings Loaded")
        # VARIANT_SCORES_DIR is where all the human gene LLRs live
        merged_df = get_LLR(reprs, VARIANT_SCORES_DIR, aa_length, args.gene)
        # the labels are already gone
        # get the missense - WT embeds
        rectified_df = get_delta_embeds(merged_df, aa_length)
        # save the output here from the delta embed process
        np.save(DELTA_EMBEDDINGS, rectified_df["repr"].values)
        print("Delta embeddings Loaded")
        # get the VAE embeddings
        X = StandardScaler().fit_transform(np.stack(rectified_df["repr"]))# actual tensors
        y = rectified_df["LLR"].to_numpy().reshape(X.shape[0], 1)
        # this trains the VAE
        if torch.cuda.is_available():
            device = "cuda"
            print("CUDA device detected, using for VAE calculations")
        else:
            device = "cpu" # make this auto detect
        # add an arg for if model_pretrained
        model = run_VAE(device, X, y) # null return -- might want to give pretrained weight arg (for whole genome sweep)
        # this fires the inference passes
        low_D_embeddings = get_low_D(model, device, rectified_df, X) 
        # save the VAE embeddings
        np.save(VAE_EMBEDDINGS, low_D_embeddings)
        print("VAE embeddings saved")
        # get the UMAP embeddings
        rectified_df = get_UMAP(low_D_embeddings, rectified_df)
        # save the UMAP embeddings
        print(rectified_df["umap_embed"].values)
        np.save(UMAP_EMBEDDINGS, rectified_df["umap_embed"].values)
        print("UMAP embeddings saved")
        # need to write out rectified_df, align later with other embeddings based on the index
        rectified_df[['gene_name', 'mutation_name', 'LLR', 
                      'clinvar_label', 'norm_LLR', 'umap1','umap2']].to_csv(RECT_DF_LOCATION, index=False)
        print("Final Dataframe saved")
    # these are from esm_VAE.py     
    else:
        # load final df
        rectified_df = pd.read_csv(RECT_DF_LOCATION)
        # load delta embeddings
        delta_embeds = np.load(DELTA_EMBEDDINGS, allow_pickle=True)
        # load VAE Embeddings
        vae_embeds = np.load(VAE_EMBEDDINGS, allow_pickle=True)
        # load UMAP Embeddings
        umap_embeds = np.load(UMAP_EMBEDDINGS, allow_pickle=True)
        # put them together on index
        rectified_df["repr"] = delta_embeds.tolist()
        rectified_df["VAE_embed"] = vae_embeds.tolist()
        rectified_df["umap_embed"] = umap_embeds.tolist()
        print("Final Dataframe with all embeddings loaded")
    # LLR ROC-AUC
    print("gene name:", dictionary[args.gene])
    # make sure there are nonzero benign and pathogenic labels
    available_labels = rectified_df.clinvar_label.unique()
    print("available_labels:", available_labels)
    if 0 in available_labels and 1 in available_labels:
        X = rectified_df[(rectified_df["clinvar_label"] == 0) | (rectified_df["clinvar_label"] == 1)]
        pred_ls = X["LLR"].to_numpy()
        y = X["clinvar_label"].to_numpy()
        print( "number of labeled variants: ", len(rectified_df.clinvar_label.dropna()))
        print("LLR ROC-AUC: " + str(roc_auc_score(y, pred_ls)))
    else:
        print( "number of labeled variants: ", len(rectified_df.clinvar_label.dropna()))
        print("LLR ROC-AUC: NaN")
    # define X and y for the esm roc-auc calc
    if len(available_labels) > 3: # NaN is included in here
        multi_class='ovr'# do not actually compute ovr
        average="mean" #was None
    else:
        multi_class='raise'
        average="mean"
    # will come out as one versus rest:
    path_name_ls = ['Benign', 'Pathogenic', 'AR-GOF', 'AR-LOF', 'AD-LOF', 'AD-GOF', 'ADAR-LOF', 'ADAR-GOF']
    # ESM
    X = np.stack(rectified_df.dropna().repr.values)
    y = rectified_df.dropna().clinvar_label.to_numpy() 
    get_all_roc(X, y, available_labels, "ESM", multi_class, average)
    # define X and y for VAE roc auc
    X = np.stack(rectified_df.dropna().VAE_embed.values)
    y = rectified_df.dropna().clinvar_label.to_numpy() 
    get_all_roc(X, y, available_labels, "VAE", multi_class, average)
    # define X and y for UMAP ROC-AUC
    X = np.stack(rectified_df.dropna().umap_embed.values)
    y = rectified_df.dropna().clinvar_label.to_numpy()     
    get_all_roc(X, y, available_labels, "UMAP", multi_class, average)    
    # these are from UMAP_graphs.py
    # LLR graph -- add more logic here
    if args.get_graphs in ["llr_only", "all"]:
        # get_pos is another input, now we use the rectified_df path_name column for labelling the graphs
        get_LLR_graph(rectified_df, args.gene, get_pos=False)
    elif args.get_graphs == "llr_pos":
        get_LLR_graph(rectified_df, args.gene, get_pos=True)
    # clinvar graph 
    # need to redo this to get the graphs with the extended label set
    if args.get_graphs in ["clinvar_only", "all"]:
        # gene_name is an input asked for before
        get_clinvar_graph(args.gene, rectified_df)
    print("All Graphs created")

def get_all_roc(X, y, path_name_in_df, roc_type, multi_class, average):    
    path_name_ls = ['Benign', 'Pathogenic', 'AR-GOF', 'AR-LOF', 'AD-LOF', 'AD-GOF', 'ADAR-LOF', 'ADAR-GOF']
    k = 0
    if multi_class == "raise":
        print(roc_type +" ROC-AUC: " + str(knn_and_roc(X, y)))       
    else:
        # need to modify knn_and_roc behavior for multiclass
        # partition into clinvar part
        if 0 in path_name_in_df and 1 in path_name_in_df:
            clinvar_X = X[(y == 0) | (y == 1)]
            clinvar_y = y[(y == 0) | (y == 1)]
            print(roc_type  + " Clinvar ROC-AUC: " +  str(knn_and_roc(clinvar_X, clinvar_y)))
        else:
            print(roc_type  + " Clinvar ROC-AUC: NaN")
        # partition into LOF-GOF part
        label_num = max(path_name_in_df[~np.isnan(path_name_in_df)])
        lof_gof_X = X[(y == label_num - 1) | (y == label_num)] 
        lof_gof_y = y[(y == label_num - 1) | (y == label_num)]
        # old version of multiclass #roc_auc_ls = knn_and_roc(X, y, multi_class, average)
        #for i in range(len(path_name_ls)):
        #    if i in path_name_in_df:
        print(roc_type  + " LOF-GOF ROC-AUC: " +  str(knn_and_roc(lof_gof_X, lof_gof_y)))
        # path_name_ls[i] +" is "+ str(roc_auc_ls[k]))
        #k += 1 
if __name__ == "__main__":
     main()
