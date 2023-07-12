# use this to actually get the AA based on some Gene ID
import numpy as np
import pandas as pd
import gzip, re
from Bio import SeqIO

three_to_one = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D',
    'Cys': 'C', 'Glu': 'E', 'Gln': 'Q', 'Gly': 'G',
    'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K',
    'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S',
    'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
}

CLINVAR_LABEL_MAPPING = {
    'Pathogenic': 1,
    'Likely pathogenic': 1,
    'Pathogenic/Likely pathogenic': 1,
    'Benign': 0,
    'Likely benign': 0,
    'Benign/Likely benign': 0,
}

import requests

def get_uniprot_id_from_genome_location(chromosome, position):
    # Construct the Ensembl REST API URL for the specified genome location
    url = f"https://rest.ensembl.org/overlap/region/human/{chromosome}:{position}:{position}?"

    # Set the desired features and content type for the API request
    params = {
        'feature': 'gene',
        'content-type': 'application/json'
    }

    # Send a GET request to the Ensembl REST API
    response = requests.get(url, params=params)

    # Extract the gene and UniProt ID from the API response
    if response.ok:
        data = response.json()
        if data:
            gene = data[0]['id']
            print(gene)
            uniprot_id = data[0]['external_name']
            print(uniprot_id)
            return gene, uniprot_id

    return None, None

def get_clinvar_df(filename):
    df = pd.read_csv(filename, sep="\t") #
    df = df[["Name", "Protein change", "Gene(s)", "Clinical significance (Last reviewed)",
            'GRCh38Chromosome','GRCh38Location']].dropna()
        #
    # will make ~ 300,000 requests
#     df['uniprot'] = df.apply(lambda row : get_uniprot_id_from_genome_location(row['GRCh38Chromosome'], row['GRCh38Location']), axis = 1)
#     print(df['uniprot'])
#     raise Error
#     df["uniprot"] = 0
#     for _, row in df.iterrows():
        
#         row["uniprot"] = get_uniprot_id_from_genome_location(row['GRCh38Chromosome'], row['GRCh38Location'])
#     print(df["uniprot"])
#     raise Error

    df["Name_edit"] = df["Name"].str.split("(p.",regex=False).str[-1].str[:-1] 
    df["Name"] = df["Name_edit"].str[:3].map(three_to_one) + df["Name_edit"].str[3:-3] + df["Name_edit"].str[-3:].map(three_to_one) 
    df = df.rename(columns = {'Protein change': 'mutation_name', 'Gene(s)': 'gene_name', 'Clinical significance (Last reviewed)':'clinvar_label'})
    df["mutation_name"] = df["mutation_name"].str.split(",") #.str[1]
    df["reviewed"] = pd.to_datetime(df['clinvar_label'].str.extract(r'Last reviewed: (\w+ \d+, \d+)')[0])
    df = (df.explode(["mutation_name"]))
    # this needs to be sped up 
    df['reviewed'] = pd.to_datetime(df['reviewed'])
    df = df.sort_values('reviewed', ascending=False).groupby(['gene_name', 'mutation_name']).first().reset_index()
#     print(df)
#     raise Error
#     # Filter rows with NaT in 'reviewed' column
#     df = df[df['reviewed'].isna()]
#     cleaned_up = []
#     for gene_name in df.gene_name:
#         smaller_df = df[(df["gene_name"] == gene_name)]
#         for mutation_name in smaller_df.mutation_name:
#             subset_df = smaller_df[(smaller_df["mutation_name"] == mutation_name)]
#             #print(subset_df)
#             #print(subset_df["reviewed"].isna().values)
#             if (subset_df["reviewed"].isna()).values[0]:
#                 cleaned_up.append(subset_df)
#             else:
#                 #print(subset_df)
#                 cleaned_up.append(subset_df.loc[subset_df['reviewed'].idxmax()])
#     df = pd.concat(cleaned_up)
    # ends here
#     print(df)
#     raise error
#     lil_d = df[df["gene_name"] == "TP53"]
#     # if a mutation has multiple labels, pick the newest one
#     print(lil_d[lil_d["mutation_name"] == "L35F"])
#     small_d = lil_d[lil_d["mutation_name"] == "L35F"]
#     print(small_d.loc[small_d['reviewed'].idxmax()])
#     raise Error
    df["mutation_name"] = df["mutation_name"].str.strip() # need to elim the empty whitespace
    #print(len(df.index))
    import re
    special_character = '*'
    # Escape the special character using re.escape
    escaped_character = re.escape(special_character)
    df = df[~df['mutation_name'].str.contains(escaped_character)]
    #print(len(df.index))
    #raise Error
    #df = df[["mutation_name", "gene_name", "clinvar_label"]]
    df["clinvar_label"] = df["clinvar_label"].str.split(r"(").str[0] 
    df["clinvar_label"] = df["clinvar_label"].map(CLINVAR_LABEL_MAPPING)
    return df

def get_uniprot(PROTEOME_FASTA_FILE_PATH):
    #DATA_DIR = 'data' # was data
    #PROTEOME_FASTA_FILE_PATH = os.path.join(DATA_DIR, 'uniprot/human_reviewed.fa.gz')
    uniprot_records = []

    with gzip.open(PROTEOME_FASTA_FILE_PATH, 'rt') as f:
        for record in SeqIO.parse(f, 'fasta'):
            gene_name = record.name.split('|')[-2] # was record.name.split('|')[-1].split('_')[0] 
            uniprot_records.append((record.name, gene_name, str(record.seq)))

    uniprot_records = pd.DataFrame(uniprot_records, columns = ['name', 'gene', 'seq'])
    gene_name_to_uniprot_records = uniprot_records.groupby('gene')
    return uniprot_records, gene_name_to_uniprot_records

def get_init_tuples(gene_list, uniprot_records, gene_name_to_uniprot_records):
    #gene_list = ["PKD1"]
    GENES_OF_INTEREST = uniprot_records[uniprot_records.gene.isin(gene_list)].gene.unique() 
    genes = []
    for gene_name in GENES_OF_INTEREST:
        (_, gene_uniprot_record), = gene_name_to_uniprot_records.get_group(gene_name).iterrows()
        seq = gene_uniprot_record['seq']
        #print('%s (%d aa)' % (gene_name, len(seq)))
        genes.append((gene_name, seq))
    return genes

def edit_and_substr(AA_str, edit):
    original_AA = edit[0]
    change_AA = edit[-1]
    location = int(edit[1:-1]) -1 # mutations are 1 indexed!
    # size of prot seq changes between revisions
    if location > len(AA_str):
        #print("original AA is only "+str(len(AA_str))+" AA not "+ str(location))
        return False
    elif AA_str[location] != original_AA: # the indexing is off by one?
        #print("original AA at pos "+str(location)+" "+original_AA+" does not match "+AA_str[location])
        return False
    #print(location, len(original_AA), AA_str[location], original_AA) 
    AA_str = AA_str[:location] + change_AA + AA_str[location+1:]
#     return center_var_context(AA_str, location)
    #  print([(AA[i], i, editted_AA[i]) for i in range(len(editted_AA)) if editted_AA[i] != AA[i]])
    return AA_str

def center_var_context(AA_str, location):
    location = int(location) -1
    ESM_CONTEXT_LEN = 1022
    # if AA_str isn't large enough
    if len(AA_str) <= ESM_CONTEXT_LEN: 
        return AA_str
    # if too early in str for center
    elif int(location) < ESM_CONTEXT_LEN/2: # pass test
        return AA_str[:ESM_CONTEXT_LEN]
    # if too late in str for center
    elif len(AA_str) - int(location) < ESM_CONTEXT_LEN/2: # pass test
        return AA_str[-ESM_CONTEXT_LEN:]    
    # if sufficient space on either side
    else: # it seems this works too now
        begin =  int(int(location)- (ESM_CONTEXT_LEN/2))
        end = int(int(location)+ (ESM_CONTEXT_LEN/2))
        #print(begin, end)
        return AA_str[begin: end]

# translating from uniprot to clinvar is the problem
def get_var_cent_tuples(genes, df):
    clinvar_muts = []
    # now create all clinvar muts
    # each of these will have a different matched WT sequence based on the windowing
    for gene, AA in genes:
        # as long as gene name appears in there, it's fine
        #matching_rows = df[df.applymap(lambda x: bool(re.search(gene, str(x)))).any(axis=1)]
        # consider that P53
        matching_rows =df[df['gene_name'] == gene] #.str.contains(fr'\b{gene}\b', regex=True, case=False)] 
       # print(matching_rows)
        # prevent duplicate mutations
        all_muts = matching_rows["mutation_name"].unique()
        all_labels = matching_rows["clinvar_label"].values
        for i in range(len(all_muts)):
            mut = all_muts[i]
            # mutations are 1 indexed!
            #print(AA)
            #print(mut)
            editted_AA = (edit_and_substr(AA, mut))
            if editted_AA is not False:
                #print(mut)
                # point mutation
                clinvar_muts.append((gene+"_"+mut, center_var_context(editted_AA, mut[1:-1])))
                # WT in frame
                clinvar_muts.append((gene+"_"+mut+"_WT", center_var_context(AA, mut[1:-1])))
    #raise Error
    return clinvar_muts




ALL_AAS = list('ACDEFGHIKLMNPQRSTVWY')
ALL_AAS_SET = set(ALL_AAS)

def get_wt_mutation(seq, mutation_names_prefix = ''):
    return (mutation_names_prefix + 'WT', seq)

def get_all_mutations(seq, includeWT = True, mutation_names_prefix = ''):
    
    seq = str(seq)
    all_mutations = []
    
    if includeWT:
        all_mutations.append(get_wt_mutation(seq, mutation_names_prefix = mutation_names_prefix))

    for pos in range(len(seq)):
        
        ref_aa = seq[pos]
        alt_aa_options = ALL_AAS_SET - {ref_aa}
        
        for alt_aa in alt_aa_options:
            mutation_name = mutation_names_prefix + ref_aa + str(pos + 1) + alt_aa
            mutation_seq = seq[:pos] + alt_aa + seq[(pos + 1):]
            all_mutations.append((mutation_name, mutation_seq))

    return all_mutations
    
def get_ESM_tuples(genes, clinvar_muts, random_size=0): # the relabeling is causing the problem
    if random_size=="full":
        all_mutations = []
        for gene_name, seq in genes:
            all_mutations.extend(get_all_mutations(seq, mutation_names_prefix = gene_name + '_'))
        return all_mutations # nadav's function
    # o.w.
    random_size = int(random_size)
    AAorder=['K','R','H','E','D','N','Q','T','S','C','G','A','V','L','I','M','P','Y','F','W']
    random_missense_addition = random_size#10000

    for gene, AA in genes:
        for i in range(random_missense_addition):
            pos = (np.random.choice(len(AA)))
            AA_pos = (np.random.choice(len(AAorder)))
            while AA[pos] == AAorder[AA_pos]:
                pos = (np.random.choice(len(AA)))
                AA_pos = (np.random.choice(len(AAorder)))
            mut = AA[pos] + str(pos +1) +AAorder[AA_pos]
            all_muts = [i[0] for i in clinvar_muts] # is this thing getting update
            if gene+"_"+mut not in all_muts:
                # construct a new mutation enry in the df, make sure to correct for the weird indexing
                # mutations are 1 indexed!
                mut = AA[pos] + str(pos +1) +AAorder[AA_pos]
                editted_AA = (edit_and_substr(AA, mut))
                if editted_AA is not False:
                    # point mutation
                    clinvar_muts.append((gene+"_"+mut, center_var_context(editted_AA, mut[1:-1])))
                    # WT in frame
                    clinvar_muts.append((gene+"_"+mut+"_WT", center_var_context(AA, mut[1:-1])))
    return clinvar_muts
