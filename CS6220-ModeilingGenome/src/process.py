import numpy as np
import pandas as pd

f_ref_seq = "../data/[Ref Genome]-MT-human.fa"
f_query_seq = "[Query-List of sequences]-MT-human_clean40.fa"
f_queries = "../data/[Query-List of sequences]-MT-human_clean40.fa"
# f_queries = "../data/[Ref Genome]-MT-human_clean4.fa"


def vec_ref_seq(ref_seq, query_list, word_len):
    query_len = len(query_list[0])
    query_df = pd.DataFrame()
    for query in query_list:
        # add_to_query_df = False  # Flag to keep track adding to the query df
        this_df = pd.DataFrame(index=[query])  # Create a data frame (one row) to store the probability of each word.
        ref_seq_i = 0
        remain = ref_seq[:]  # Copies the ref seq into the remaining sequence.
        query_i = remain.find(query)
        # if there is nothing to add to the query_df continue to the next entry.
        if not (query_i != -1 and len(remain[query_i:]) >= word_len + query_len):
            continue
        while query_i != -1 and len(remain[query_i:]) >= word_len + query_len:  # Query sequence exists with room for word
            remain = remain[query_i:]
            word = remain[query_len: query_len + word_len]
            if word in this_df:
                this_df[word] += 1
            else:
                this_df[word] = 1
            remain = remain[1:]
            query_i = remain.find(query)
        this_df = this_df / this_df.values.sum()
        query_df = pd.concat([query_df, this_df], sort=False)
    query_df = query_df.fillna(value=0)

    return query_df


def load_ref_seq():
    with open(f_ref_seq, "r", newline='') as f:
        my_ref_seq = f.read().replace("\n", "")
    return my_ref_seq


def load_queries():
    with open(f_queries, "r", newline='') as f:
        my_query_list = f.read().split()
    return my_query_list


def main():
    # demo()

    query_list = load_queries()
    # Trim query list
    trim_query_len = 6
    for i, query in enumerate(query_list):
        query_list[i] = query_list[i][:trim_query_len]


    reference_sequence = load_ref_seq()
    vectorized_ref = vec_ref_seq(reference_sequence, query_list, 2)



def demo():
    ref_seq_test = "ATCGTATCGTACTGACTGATC"
    query_list_test = ["CGT", "ACT", "TAT"]
    word_len_test = 2
    print(vec_ref_seq(ref_seq_test, query_list_test, word_len_test))


if __name__ == '__main__':
    main()





