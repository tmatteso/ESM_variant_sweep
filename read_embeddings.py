import glob, os, torch
import pandas as pd
import numpy as np
# variant centered approach: just create a window with the desired variant in the center
# one window per missense mutation, get per token embeddings for this sequence
# 1022/2 = 511 (mutation pos)
# also get the mutation centered WT window, 
# subtract the WT per token embedding - missense per token embedding
# then get the mean representation to (1028,)
# doesn't have to be full missense variant set, just some >1000 size subset


def read_esm_reprs(all_embeddings, df):
    # esm_embeddings_per_token has the mean representations so you can do something fast for Nadav
    # but to actually get to the next step means you need LLRs for the VAE
    all_embeddings = glob.glob(all_embeddings +"/*.pt") #new_esm_embeddings
    # refactor this to do the esm1b embed subtraction sooner
    # something happened here
    row_ls = []
    for embedding in all_embeddings:        
        possible_outputs = embedding.split("/")[-1].split(".")[0].split("_") 
        if len(possible_outputs) == 2:
            gene_name, mutation_name = possible_outputs # tuple unpack
            WT = False
        elif len(possible_outputs) == 3: # all the new benign gor read in strangely
            gene_name, mutation_name, WT = possible_outputs 
            WT = True# override WT to a boolean
        elif len(possible_outputs) == 1:
            gene_name = possible_outputs[0]
            mutation_name = "WT"
            WT = True
        else:
            print(possible_outputs)
            raise ValueError
        # not all will have a clinvar label -- you need the whole df in mem to remember the labels!
        regex_pattern = r'^' + mutation_name + r'$'
        lil_d = df[df['mutation_name'].str.match(regex_pattern)]
        #lil_d = df[df["mutation_name"] == mutation_name]
        # this is returning way too many nans
        # must be a problem with gene_name
    #     raise Error
#         if mutation_name[0:4] == "R273":
#             print(mutation_name)
        if len(lil_d.index) == 1:
            label = lil_d["clinvar_label"].values[0]
            path_name = lil_d["path_name"].values[0]
        elif len(lil_d.index) == 2: # these means the extra labels overrode clinvar
            label = lil_d["clinvar_label"].values[1]   
            path_name = lil_d["path_name"].values[1]
        else:
            label = np.NaN
            path_name=""
        # I think adding this directly to the df will be too slow at scale
        # just leave path to file
        window_mutation_seq = embedding
        
        #torch.load(embedding)["representations"][33].numpy() 
        #print(window_mutation_seq.keys()) # per token only gives references, not logits
        # this is also not a scalable strategy -- you would have to calc each of these individually
        #["representations"][33].numpy()
        #["mean_representations"][33].numpy() # mean reps
        #torch.load(embedding)["representations"][33].numpy()# #per token representations
        row_ls.append([gene_name, mutation_name, window_mutation_seq, label, path_name, WT])

    reprs = pd.DataFrame(columns=['gene_name', 'mutation_name', 'mutation_repr', 'clinvar_label', 'path_name', 'WT'], data=row_ls) 
    return reprs

# make this for all genes
# let's just get the uniprot names directly
to_uniprot = {
        "PKD1":"P98161",
        "P53":"P04637"
    }



def get_LLR(reprs, VARIANT_SCORES_DIR, aa_length, gene_name): # clinvar labels lost
    #VARIANT_SCORES_DIR = "data/esm1b_variant_scores/ALL_hum_isoforms_ESM1b_LLR"
    #relevant_LLR = []
    #print(reprs.clinvar_label.unique())
    #for gene_name in reprs['gene_name'].unique(): # '%s_LLR.csv' % to_uniprot[gene_name])
    #    if gene_name in to_uniprot.keys():
    #        gene_name = to_uniprot[gene_name]
        #print(gene_name)
        # the WT is lost here, look at your newer code
    raw_LLR_df = pd.read_csv(os.path.join(VARIANT_SCORES_DIR, '%s_LLR.csv' %  gene_name), index_col = 0)
    #print(raw_LLR_df)
    LLR_df = raw_LLR_df.stack().reset_index().rename(columns = {'level_0': 'mt_aa', 'level_1': 'wt_aa_and_pos', 0: 'LLR'})
    LLR_df['mutation_name'] = LLR_df['wt_aa_and_pos'].str.replace(' ', '') + LLR_df['mt_aa']
    del LLR_df['wt_aa_and_pos'], LLR_df['mt_aa']
    LLR_df['gene_name'] = gene_name
    #relevant_LLR.append(LLR_df)
    # now add WT
    if aa_length <= 1022:
        dic = {'LLR': 0.0, 'mutation_name': "WT", 'gene_name': "P53"}
        LLR_df = pd.concat([LLR_df, pd.DataFrame([dic])], ignore_index=True)
    # now merge relevant_LLR and reprs on mutation name
    #print(pd.merge(reprs, LLR_df, on="mutation_name").clinvar_label.unique())
    return pd.merge(reprs, LLR_df, on="mutation_name")

def get_mut_pos_in_embed(mutation, aa_length):
    pos = int(mutation[1:-1]) -1
    ESM_CONTEXT_LEN = 1022
    if aa_length <= ESM_CONTEXT_LEN:
        return pos
        # if too early in str for center
    elif pos < ESM_CONTEXT_LEN/2: 
        return pos
    # if too late in str for center
    elif aa_length - pos < ESM_CONTEXT_LEN/2: # pass test
        #print(-1*(aa_length - pos))
        return int(-1*(aa_length - pos)) # this would be the negative index
    # if sufficient space on either side
    else: # it seems this works too now
        # then should be dead center
        return int(ESM_CONTEXT_LEN/2)

# new function
# take the embedding just from the mutaton position and ignore all other positions: 
# just slice on the index, perform the subtraction, and do not take the average, then recompute all embeddings and scores
def get_delta_on_mut_pos(merged_df, aa_length):
    rectified_df = []
    print(merged_df.gene_name_x.unique())
    # have to go through the vector subtraction step here, elims the WT
    for gene in merged_df.gene_name_x.unique():
        for mut in merged_df.mutation_name.unique():
            #print(gene, mut)
            if aa_length <= 1022:
                # then the WT is different, doesn't need to do variant centering 
                WT = merged_df[(merged_df["WT"] == True)]["mutation_repr"].values[0]
            else:
                # gene_name might be an issue again
                WT = merged_df[(merged_df["gene_name_x"] == gene) &
                               (merged_df["mutation_name"] == mut) &
                               (merged_df["WT"] == True)]["mutation_repr"].values[0]
            if mut == "WT":
                # check that the big one with full to make sure it make sure it's alright
                missense = merged_df[(merged_df["gene_name_x"] == gene) &
                                     (merged_df["mutation_name"] == mut) &
                                     (merged_df["WT"] == True)]["mutation_repr"].values[0]
                LLR = merged_df[(merged_df["gene_name_x"] == gene) &
                                (merged_df["mutation_name"] == mut) &
                                (merged_df["WT"] == True)]["LLR"].values[0]
                clinvar_label = merged_df[(merged_df["gene_name_x"] == gene) &
                                (merged_df["mutation_name"] == mut) &
                                (merged_df["WT"] == True)]["clinvar_label"].values[0]
            else:
                missense = merged_df[(merged_df["gene_name_x"] == gene) &
                                     (merged_df["mutation_name"] == mut) &
                                     (merged_df["WT"] == False)]["mutation_repr"].values[0]
                LLR = merged_df[(merged_df["gene_name_x"] == gene) &
                                (merged_df["mutation_name"] == mut) &
                                (merged_df["WT"] == False)]["LLR"].values[0]
                clinvar_label = merged_df[(merged_df["gene_name_x"] == gene) &
                                (merged_df["mutation_name"] == mut) &
                                (merged_df["WT"] == False)]["clinvar_label"].values[0]
            WT_embed = torch.load(WT)["representations"][33].numpy()
            missense_embed = torch.load(missense)["representations"][33].numpy()
            pos = get_mut_pos_in_embed(mut, aa_length)
            #print(mut, aa_length, pos)
            # now use the aa_len and the mut position to slice the embedding
            delta_embed = WT_embed[pos] - missense_embed[pos] # check shape of these embeds before indexing
            rectified_df.append([gene,
                                mut,
                                delta_embed, #performs the sum-- do not sum for this version
                                LLR,
                                clinvar_label])
    rectified_df = pd.DataFrame(columns=['gene_name', 'mutation_name', 'repr', 'LLR', 'clinvar_label'],
                                data=rectified_df)

    rectified_df['norm_LLR'] = np.clip(rectified_df['LLR'], None, 0)
    rectified_df['norm_LLR'] -= rectified_df['norm_LLR'].min()
    rectified_df['norm_LLR'] /= rectified_df['norm_LLR'].max()
    return rectified_df

def get_delta_embeds(merged_df, aa_length):
    rectified_df = []
    print(merged_df.gene_name_x.unique())
    # have to go through the vector subtraction step here, elims the WT
    for gene in merged_df.gene_name_x.unique():
        for mut in merged_df.mutation_name.unique():
            #print(gene, mut)
            if aa_length <= 1022:
                # then the WT is different, doesn't need to do variant centering 
                WT = merged_df[(merged_df["WT"] == True)]["mutation_repr"].values[0]
            else:
                # gene_name might be an issue again
                WT = merged_df[(merged_df["gene_name_x"] == gene) &
                               (merged_df["mutation_name"] == mut) & 
                               (merged_df["WT"] == True)]["mutation_repr"].values[0]
            if mut == "WT":
                # check that the big one with full to make sure it make sure it's alright
                missense = merged_df[(merged_df["gene_name_x"] == gene) &
                                     (merged_df["mutation_name"] == mut) &
                                     (merged_df["WT"] == True)]["mutation_repr"].values[0]            
                LLR = merged_df[(merged_df["gene_name_x"] == gene) &
                                (merged_df["mutation_name"] == mut) & 
                                (merged_df["WT"] == True)]["LLR"].values[0]
                clinvar_label = merged_df[(merged_df["gene_name_x"] == gene) &
                                (merged_df["mutation_name"] == mut) & 
                                (merged_df["WT"] == True)]["clinvar_label"].values[0]
            else:
                missense = merged_df[(merged_df["gene_name_x"] == gene) &
                                     (merged_df["mutation_name"] == mut) & 
                                     (merged_df["WT"] == False)]["mutation_repr"].values[0]
                LLR = merged_df[(merged_df["gene_name_x"] == gene) &
                                (merged_df["mutation_name"] == mut) & 
                                (merged_df["WT"] == False)]["LLR"].values[0]
                clinvar_label = merged_df[(merged_df["gene_name_x"] == gene) &
                                (merged_df["mutation_name"] == mut) & 
                                (merged_df["WT"] == False)]["clinvar_label"].values[0]
            WT_embed = torch.load(WT)["representations"][33].numpy() 
            missense_embed = torch.load(missense)["representations"][33].numpy() 
            delta_embed = WT_embed - missense_embed 
            rectified_df.append([gene,
                                mut,
                                delta_embed.sum(axis = 0), #performs the sum
                                LLR,
                                clinvar_label])
    rectified_df = pd.DataFrame(columns=['gene_name', 'mutation_name', 'repr', 'LLR', 'clinvar_label'],
                                data=rectified_df)

    rectified_df['norm_LLR'] = np.clip(rectified_df['LLR'], None, 0)
    rectified_df['norm_LLR'] -= rectified_df['norm_LLR'].min()
    rectified_df['norm_LLR'] /= rectified_df['norm_LLR'].max()
    return rectified_df
