#!/bin/bash
echo 'gene name,available_labels,number of labeled variants,LLR ROC-AUC,ESM Clinvar ROC-AUC,ESM LOF-GOF ROC-AUC,VAE Clinvar ROC-AUC,VAE LOF-GOF ROC-AUC,UMAP Clinvar ROC-AUC,UMAP LOF-GOF ROC-AUC' > output.csv
grep -E 'gene name|available_labels|number of labeled variants|LLR ROC-AUC|ESM Clinvar ROC-AUC|ESM LOF-GOF ROC-AUC|VAE Clinvar ROC-AUC|VAE LOF-GOF ROC-AUC|UMAP Clinvar ROC-AUC|UMAP LOF-GOF ROC-AUC' output.txt | awk 'BEGIN {FS = ": "} {print $2}' | paste -d "," - - - - - - - - - ->> output.csv
