import pandas as pd
from io import StringIO

# Your provided standard output
std_output = """
LLR Benign Pathogenic avg ROC-AUC: 0.12120725034110558
LLR-kNN Benign Pathogenic avg ROC-AUC: 0.7962369189402863
ESM Benign Pathogenic kNN avg ROC-AUC: 0.7254228602342039
ESM Benign Pathogenic Gaussian NB avg ROC-AUC: 0.7505754171322023
ESM Benign Pathogenic Gaussian NB avg ROC-AUC: 0.5423457649675126
ESM Benign Pathogenic Gaussian NB avg ROC-AUC: 0.6201385015668249
ESM Benign Pathogenic Random Forest avg ROC-AUC: 0.5240374770875512
ESM Benign Pathogenic Random Forest avg ROC-AUC: 0.5998016995104074
ESM Benign Pathogenic Random Forest avg ROC-AUC: 0.5
ESM Benign Pathogenic Logistic Regression avg ROC-AUC: 0.803377987607394
ESM Benign Pathogenic Logistic Regression avg ROC-AUC: 0.6761523654135011
ESM Benign Pathogenic Logistic Regression avg ROC-AUC: 0.5292415332439154
VAE Benign Pathogenic kNN avg ROC-AUC: 0.6920437230617322
VAE Benign Pathogenic Gaussian NB avg ROC-AUC: 0.7505754171322023
VAE Benign Pathogenic Gaussian NB avg ROC-AUC: 0.5423457649675126
VAE Benign Pathogenic Gaussian NB avg ROC-AUC: 0.6201385015668249
VAE Benign Pathogenic Random Forest avg ROC-AUC: 0.5240374770875512
VAE Benign Pathogenic Random Forest avg ROC-AUC: 0.5998016995104074
VAE Benign Pathogenic Random Forest avg ROC-AUC: 0.5
VAE Benign Pathogenic Logistic Regression avg ROC-AUC: 0.803377987607394
VAE Benign Pathogenic Logistic Regression avg ROC-AUC: 0.6761523654135011
VAE Benign Pathogenic Logistic Regression avg ROC-AUC: 0.5292415332439154
UMAP Benign Pathogenic kNN avg ROC-AUC: 0.813002220215678
UMAP Benign Pathogenic Gaussian NB avg ROC-AUC: 0.7505754171322023
UMAP Benign Pathogenic Gaussian NB avg ROC-AUC: 0.5423457649675126
UMAP Benign Pathogenic Gaussian NB avg ROC-AUC: 0.6201385015668249
UMAP Benign Pathogenic Random Forest avg ROC-AUC: 0.5240374770875512
UMAP Benign Pathogenic Random Forest avg ROC-AUC: 0.5998016995104074
UMAP Benign Pathogenic Random Forest avg ROC-AUC: 0.5
UMAP Benign Pathogenic Logistic Regression avg ROC-AUC: 0.803377987607394
UMAP Benign Pathogenic Logistic Regression avg ROC-AUC: 0.6761523654135011
UMAP Benign Pathogenic Logistic Regression avg ROC-AUC: 0.5292415332439154
LLR AR-GOF AR-LOF avg ROC-AUC: 0.13333333333333333
LLR-kNN AR-GOF AR-LOF avg ROC-AUC: 0.5
ESM AR-GOF AR-LOF kNN avg ROC-AUC: 0.5
ESM AR-GOF AR-LOF Gaussian NB avg ROC-AUC: 0.6333333333333332
ESM AR-GOF AR-LOF Gaussian NB avg ROC-AUC: 0.4666666666666666
ESM AR-GOF AR-LOF Gaussian NB avg ROC-AUC: 0.6666666666666666
ESM AR-GOF AR-LOF Random Forest avg ROC-AUC: 0.9
ESM AR-GOF AR-LOF Random Forest avg ROC-AUC: 1.0
ESM AR-GOF AR-LOF Random Forest avg ROC-AUC: 0.9666666666666666
ESM AR-GOF AR-LOF Logistic Regression avg ROC-AUC: 0.8333333333333333
ESM AR-GOF AR-LOF Logistic Regression avg ROC-AUC: 0.9
ESM AR-GOF AR-LOF Logistic Regression avg ROC-AUC: 0.8666666666666668
VAE AR-GOF AR-LOF kNN avg ROC-AUC: 0.5
VAE AR-GOF AR-LOF Gaussian NB avg ROC-AUC: 0.6333333333333332
VAE AR-GOF AR-LOF Gaussian NB avg ROC-AUC: 0.4666666666666666
VAE AR-GOF AR-LOF Gaussian NB avg ROC-AUC: 0.6666666666666666
VAE AR-GOF AR-LOF Random Forest avg ROC-AUC: 0.9
VAE AR-GOF AR-LOF Random Forest avg ROC-AUC: 1.0
VAE AR-GOF AR-LOF Random Forest avg ROC-AUC: 0.9666666666666666
VAE AR-GOF AR-LOF Logistic Regression avg ROC-AUC: 0.8333333333333333
VAE AR-GOF AR-LOF Logistic Regression avg ROC-AUC: 0.9
VAE AR-GOF AR-LOF Logistic Regression avg ROC-AUC: 0.8666666666666668
UMAP AR-GOF AR-LOF kNN avg ROC-AUC: 0.5
UMAP AR-GOF AR-LOF Gaussian NB avg ROC-AUC: 0.6333333333333332
UMAP AR-GOF AR-LOF Gaussian NB avg ROC-AUC: 0.4666666666666666
UMAP AR-GOF AR-LOF Gaussian NB avg ROC-AUC: 0.6666666666666666
UMAP AR-GOF AR-LOF Random Forest avg ROC-AUC: 0.9
UMAP AR-GOF AR-LOF Random Forest avg ROC-AUC: 1.0
UMAP AR-GOF AR-LOF Random Forest avg ROC-AUC: 0.9666666666666666
UMAP AR-GOF AR-LOF Logistic Regression avg ROC-AUC: 0.8333333333333333
UMAP AR-GOF AR-LOF Logistic Regression avg ROC-AUC: 0.9
UMAP AR-GOF AR-LOF Logistic Regression avg ROC-AUC: 0.8666666666666668
LLR AD-LOF AD-GOF avg ROC-AUC: 0.6937951516919593
LLR-kNN AD-LOF AD-GOF avg ROC-AUC: 0.5986433659775027
ESM AD-LOF AD-GOF kNN avg ROC-AUC: 0.5333131717550942
ESM AD-LOF AD-GOF Gaussian NB avg ROC-AUC: 0.6036872918772331
ESM AD-LOF AD-GOF Gaussian NB avg ROC-AUC: 0.5673216285661309
ESM AD-LOF AD-GOF Gaussian NB avg ROC-AUC: 0.5831178705274851
ESM AD-LOF AD-GOF Random Forest avg ROC-AUC: 0.5913045305409911
ESM AD-LOF AD-GOF Random Forest avg ROC-AUC: 0.5968280200285102
ESM AD-LOF AD-GOF Random Forest avg ROC-AUC: 0.5617459117166982
ESM AD-LOF AD-GOF Logistic Regression avg ROC-AUC: 0.7753097305721094
ESM AD-LOF AD-GOF Logistic Regression avg ROC-AUC: 0.7849846651547745
ESM AD-LOF AD-GOF Logistic Regression avg ROC-AUC: 0.7889470033848613
VAE AD-LOF AD-GOF kNN avg ROC-AUC: 0.6521416277313957
VAE AD-LOF AD-GOF Gaussian NB avg ROC-AUC: 0.6036872918772331
VAE AD-LOF AD-GOF Gaussian NB avg ROC-AUC: 0.5673216285661309
VAE AD-LOF AD-GOF Gaussian NB avg ROC-AUC: 0.5831178705274851
VAE AD-LOF AD-GOF Random Forest avg ROC-AUC: 0.5913045305409911
VAE AD-LOF AD-GOF Random Forest avg ROC-AUC: 0.5968280200285102
VAE AD-LOF AD-GOF Random Forest avg ROC-AUC: 0.5617459117166982
VAE AD-LOF AD-GOF Logistic Regression avg ROC-AUC: 0.7753097305721094
VAE AD-LOF AD-GOF Logistic Regression avg ROC-AUC: 0.7849846651547745
VAE AD-LOF AD-GOF Logistic Regression avg ROC-AUC: 0.7889470033848613
UMAP AD-LOF AD-GOF kNN avg ROC-AUC: 0.7825255975587228
UMAP AD-LOF AD-GOF Gaussian NB avg ROC-AUC: 0.6036872918772331
UMAP AD-LOF AD-GOF Gaussian NB avg ROC-AUC: 0.5673216285661309
UMAP AD-LOF AD-GOF Gaussian NB avg ROC-AUC: 0.5831178705274851
UMAP AD-LOF AD-GOF Random Forest avg ROC-AUC: 0.5913045305409911
UMAP AD-LOF AD-GOF Random Forest avg ROC-AUC: 0.5968280200285102
UMAP AD-LOF AD-GOF Random Forest avg ROC-AUC: 0.5617459117166982
UMAP AD-LOF AD-GOF Logistic Regression avg ROC-AUC: 0.7753097305721094
UMAP AD-LOF AD-GOF Logistic Regression avg ROC-AUC: 0.7849846651547745
UMAP AD-LOF AD-GOF Logistic Regression avg ROC-AUC: 0.7889470033848613
LLR ADAR-LOF ADAR-GOF avg ROC-AUC: 0.6631806649398367
LLR-kNN ADAR-LOF ADAR-GOF avg ROC-AUC: 0.49212227919705454
ESM ADAR-LOF ADAR-GOF kNN avg ROC-AUC: 0.5748953175621981
ESM ADAR-LOF ADAR-GOF Gaussian NB avg ROC-AUC: 0.6533209293559101
ESM ADAR-LOF ADAR-GOF Gaussian NB avg ROC-AUC: 0.5618671737975331
ESM ADAR-LOF ADAR-GOF Gaussian NB avg ROC-AUC: 0.613430072993616
ESM ADAR-LOF ADAR-GOF Random Forest avg ROC-AUC: 0.5161407760332665
ESM ADAR-LOF ADAR-GOF Random Forest avg ROC-AUC: 0.5114324566105696
ESM ADAR-LOF ADAR-GOF Random Forest avg ROC-AUC: 0.4931102772566187
ESM ADAR-LOF ADAR-GOF Logistic Regression avg ROC-AUC: 0.6493177322606078
ESM ADAR-LOF ADAR-GOF Logistic Regression avg ROC-AUC: 0.5672571829824719
ESM ADAR-LOF ADAR-GOF Logistic Regression avg ROC-AUC: 0.516321751454935
VAE ADAR-LOF ADAR-GOF kNN avg ROC-AUC: 0.615172928804187
VAE ADAR-LOF ADAR-GOF Gaussian NB avg ROC-AUC: 0.6533209293559101
VAE ADAR-LOF ADAR-GOF Gaussian NB avg ROC-AUC: 0.5618671737975331
VAE ADAR-LOF ADAR-GOF Gaussian NB avg ROC-AUC: 0.613430072993616
VAE ADAR-LOF ADAR-GOF Random Forest avg ROC-AUC: 0.5161407760332665
VAE ADAR-LOF ADAR-GOF Random Forest avg ROC-AUC: 0.5114324566105696
VAE ADAR-LOF ADAR-GOF Random Forest avg ROC-AUC: 0.4931102772566187
VAE ADAR-LOF ADAR-GOF Logistic Regression avg ROC-AUC: 0.6493177322606078
VAE ADAR-LOF ADAR-GOF Logistic Regression avg ROC-AUC: 0.5672571829824719
VAE ADAR-LOF ADAR-GOF Logistic Regression avg ROC-AUC: 0.516321751454935
UMAP ADAR-LOF ADAR-GOF kNN avg ROC-AUC: 0.5278510241363258
UMAP ADAR-LOF ADAR-GOF Gaussian NB avg ROC-AUC: 0.6533209293559101
UMAP ADAR-LOF ADAR-GOF Gaussian NB avg ROC-AUC: 0.5618671737975331
UMAP ADAR-LOF ADAR-GOF Gaussian NB avg ROC-AUC: 0.613430072993616
UMAP ADAR-LOF ADAR-GOF Random Forest avg ROC-AUC: 0.5161407760332665
UMAP ADAR-LOF ADAR-GOF Random Forest avg ROC-AUC: 0.5114324566105696
UMAP ADAR-LOF ADAR-GOF Random Forest avg ROC-AUC: 0.4931102772566187
UMAP ADAR-LOF ADAR-GOF Logistic Regression avg ROC-AUC: 0.6493177322606078
UMAP ADAR-LOF ADAR-GOF Logistic Regression avg ROC-AUC: 0.5672571829824719
UMAP ADAR-LOF ADAR-GOF Logistic Regression avg ROC-AUC: 0.516321751454935
"""

# Read the standard output into a DataFrame
data = StringIO(std_output)
df = pd.read_csv(data, sep=':', header=None, names=['Model', 'Metrics'])

# Extract the metrics values
df['Metrics'] = df['Metrics'].astype(str)
print(df)
raise Error
df['Metrics'] = df['Metrics'].str.strip()
df[['Metric', 'Value']] = df['Metrics'].str.extract(r'([\w\s-]+)ROC-AUC:\s(.*)')

# Pivot the DataFrame to get the desired format
pivot_df = df.pivot(index='Model', columns='Metric', values='Value')

# Define the order of columns and rows
columns_order = ['Pathogenic Benign', 'AR-LOF AR-GOF', 'AD-LOF AD-GOF', 'ADAR-LOF ADAR-GOF']
rows_order = ['LLR', 'LLR-kNN', 'ESM kNN', 'ESM Gaussian NB', 'ESM Random Forest', 'ESM Logistic Regression',
              'VAE kNN', 'VAE Gaussian NB', 'VAE Random Forest', 'VAE Logistic Regression',
              'UMAP k-NN', 'UMAP Gaussian NB', 'UMAP Random Forest', 'UMAP Logistic Regression']

# Reorder columns and rows
pivot_df = pivot_df[columns_order].reindex(index=rows_order)

# Write the DataFrame to a CSV file
output_csv = 'full_sweep_output.csv'
pivot_df.to_csv(output_csv)

print(f'CSV file "{output_csv}" has been created.')

