#!/bin/bash
#P53_embeds/ 

# names=("P42224" "Q01959" "Q8NBP7" "P40763" "P35498" "P07949" "Q14654" "P00441" "P63252")
# run them all again
names=(Q14524 P51787 Q09428 P41180 P04275 P29033 Q12809 P16473 O43526 Q01959 P40763 P35498 Q14654 Q8NBP7 P07949 P42224 P00441 P63252)
#names=("P42224" "Q01959" "Q8NBP7" "P40763" "P35498" "P07949" "Q14654" "P00441" "P63252")
for name in "${names[@]}"
do
    python3 query.py --gene="$name" --num_random=1000 --df_preloaded=True --existing_embeddings=all --extra_labels="variants (1).csv" >> output.txt
done
#python3 query.py --gene=Q12809 --num_random=1000 --df_preloaded=False --existing_embeddings=None --extra_labels=variants\ \(1\).csv
