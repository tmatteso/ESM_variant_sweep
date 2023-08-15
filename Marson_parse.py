import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def import_and_explode(protein_location, all_data_location):
    # import required dfs
    protein = pd.read_csv(protein_location, sep="\t")
    # base editor NGG I believe with four sorts (high and low): IFNy, PD1, TNFa, CD25
    all_data = pd.read_csv(all_data_location, sep="\t")
    # explode only on sites
    separate_sites = all_data[["gene", "marker", "sites", "LFC"]]
    separate_sites["sites"] = separate_sites["sites"].str[1:-1].str.split(", ")
    separate_sites = separate_sites.explode('sites')
    separate_sites['sites'] = separate_sites['sites'].str[1:-1]
    return separate_sites, protein

def norm_LFC(separate_sites):
    # normalize the LFC for all variants of a gene
    gene_set = separate_sites.gene.unique()
    new_df = []
    for gene in gene_set:
        for marker in separate_sites[(separate_sites.gene == gene)].marker.unique():
            # normalize the LFC for all variants of a gene, divide by the s.d, no mean centering!
            subset = separate_sites[(separate_sites.gene == gene) & (separate_sites.marker == marker)]
            st_dev = np.std(subset.LFC.values)
            # broadcast the divison and create a new row, norm_LFC
            subset["norm_LFC"] = subset.LFC.values / st_dev
            new_df.append(subset)
    # save the gene	marker	sites	LFC norm_LFC LLR df        
    new_df = pd.concat(new_df)
    
    return new_df

def elim_non_misense(new_df):
    # eliminate splice variants and any other non-missense variants
    new_df = new_df.dropna()
    new_df = new_df[~new_df['sites'].str.contains('splice')]
    new_df = new_df[new_df['sites'].str.match(r'^[A-Za-z]\d+[A-Za-z]$')] # this will also remove any *123A or A123* type vars
    return new_df

# must have sites col in the df
def center_aa_frame(with_aa_seq, col_to_apply, new_col):
    # Define the ESM_CONTEXT_LEN constant
    ESM_CONTEXT_LEN = 1022
    # Create a lambda function for center_var_context
    center_var_context_lambda = lambda row: (
        row[col_to_apply] if len(row[col_to_apply]) <= ESM_CONTEXT_LEN else
        row[col_to_apply][:ESM_CONTEXT_LEN] if int(row['sites'][1:-1]) < ESM_CONTEXT_LEN/2 else
        row[col_to_apply][-ESM_CONTEXT_LEN:] if len(row[col_to_apply]) - int(row['sites'][1:-1]) < ESM_CONTEXT_LEN/2 else
        row[col_to_apply][int(int(row['sites'][1:-1]) - (ESM_CONTEXT_LEN/2)): int(int(row['sites'][1:-1]) + (ESM_CONTEXT_LEN/2))]
    )
    # Apply the lambda function to create a new 'centered_var_context' column
    with_aa_seq[new_col] = with_aa_seq.apply(center_var_context_lambda, axis=1)
    return with_aa_seq

def execute_missense(with_aa_seq):
    # for each site, apply the change to the protein_seq column
    with_aa_seq['mod_seq'] = with_aa_seq.apply(
        lambda row: row['protein_seq'][:int(row['sites'][1:-1])] +  # get the 511
        row['sites'][-1] + # this what to sub
        row['protein_seq'][int(row['sites'][1:-1]) +1:], 
        axis=1) 
    return with_aa_seq

# fasta prep
def format_fasta_entry(row):
    fasta_id = f"{row['fasta_id']}"
    # f"{row['gene']}_{row['marker']}_{row['sites']}"
    return f">{fasta_id}\n{row['centered_var_context']}"

def transform_row(row):
    rows = [
        {'fasta_id': row['fasta_id'], 'centered_var_context': row['centered_var_context'], 'norm_LFC': row['norm_LFC']},
        {'fasta_id': row['fasta_id'] + '_WT', 'centered_var_context': row['centered_WT_context'], 'norm_LFC': row['norm_LFC']}
    ]
    return rows

def sep_WT(with_aa_seq):
    # Create a lambda function
    with_aa_seq["fasta_id"] = with_aa_seq.apply(
        lambda row: f"{row['gene']}_{row['marker']}_{row['sites']}",
        axis=1
    )
    # take average of duplicate fasta_id names
    fasta_subset = with_aa_seq.groupby(['fasta_id', 'centered_var_context', 'centered_WT_context'],
                                       as_index=False)['norm_LFC'].mean()
    # now separate the WT, making new fasta_id rows 
    # Apply the transformation using the lambda function
    transformed_rows = fasta_subset.apply(lambda row: transform_row(row), axis=1)
    # Create the transformed DataFrame
    transformed_df = pd.DataFrame([item for sublist in transformed_rows for item in sublist])
    # it keeps them together, WT right after the mutation!
    return transformed_df

    
def make_fastas(with_aa_seq):
    # Iterate through the DataFrame and create FASTA entries
    fasta_entries = with_aa_seq.apply(format_fasta_entry, axis=1)
    # chunk 50,000 seqs at a time
    chunk_size = 50000
    for start_index in range(0, len(fasta_entries), chunk_size):
        end_index = start_index + chunk_size
        split_series = fasta_entries.iloc[start_index:end_index]
        # Write the FASTA entries to a file
        fasta_file_path = 'protein_sequences_'+ str(start_index) + '.fasta'
        with open(fasta_file_path, 'w') as fasta_file:
            fasta_file.write('\n'.join(split_series))
    # write out this df as well with only fasta_id  norm_LFC remaining
    with_aa_seq[["fasta_id", "norm_LFC"]].to_csv("Marson_LFC.csv")
    
def main():
    # all_data, you need seq too, the guide rna is different, do not get rid of that column. Remove duplicates after inclusing seq. If you don't see duplicates, then you should be fine. 
    
    protein_location = "BE026_protein_data (1).txt"
    all_data_location = "BE026_all_data (1).txt"
    separate_sites, protein = import_and_explode(protein_location, all_data_location)
    new_df = norm_LFC(separate_sites)
    # elim splice donor/acceptors, other weirdness
    new_df = elim_non_misense(new_df)
    # join on gene, give new_df the desired amino acid sequence
    with_aa_seq = pd.merge(new_df, protein[["gene", "protein_seq"]], on='gene', how='left')
    # ignore all variants/WT for PTEN, that gene has many stop codons. 
    with_aa_seq = with_aa_seq[with_aa_seq["gene"] != "PTEN"]
    # make the missense happen
    with_aa_seq = execute_missense(with_aa_seq)
    # get var centered muts
    col_to_apply, new_col = "mod_seq", "centered_var_context"
    with_aa_seq = center_aa_frame(with_aa_seq, col_to_apply, new_col)
    # get var centered WTs
    col_to_apply, new_col = "protein_seq", "centered_WT_context"
    # this will make duplicate rows on fasta_id col
    with_aa_seq = center_aa_frame(with_aa_seq, col_to_apply, new_col)
    # make sure you take the last char out if char == *
    transformed_df = sep_WT(with_aa_seq)
    # at first instance of a stop codon, chop the rest of the sequence -- do for both mut and WT
    transformed_df['centered_var_context'] = transformed_df['centered_var_context'].str.replace(r'\*.*', '', regex=True)
    # by defn, this will leave no WT-missense orphans then.
    # finish, make fastas with 50,000 seqs at a time
    make_fastas(transformed_df)
    
if __name__ == "__main__":
    main()