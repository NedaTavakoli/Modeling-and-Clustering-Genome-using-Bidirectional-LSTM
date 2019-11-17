import numpy as np
import pandas as pd
import csv

f_queries = "../data/[Query-List of sequences]-MT-human_clean40.fa"
f_embeddings = "../vector_representation/[Query-List of sequences]-MT-human_clean40_genome_embeddings.csv"

# filename = "test.csv"


def load_queries():
    with open(f_queries, "r", newline='') as f:
        my_query_list = f.read().split()
    return my_query_list

print(query_list)

my_df = pd.read_csv(f_embeddings, names=np.arange(1000))



print(my_df.shape)
print(len(query_list))