# Modeling-and-Clustering-Genome-using-Bidirectional-LSTM
Read Me:

We ran the experiments for the following configurations. At each configuration, we model a reference genome and a list of query sequences. Two types of outputs are reported: vector representations (after training the model using LSTM), and perplexity of each epoch of training.


Configuration 1:
	len data:  158970136 characters  (data/Ref Genome]-MT-human2.fa)
	word length =  2
    batch_size=100
	hidden_size=1000
	initial_learning rate=0.001
	num_epochs=100
	num_layers=1
	sequence size=24

	Input dataset:
		Ref genome: data/Ref/MT-human_clean2.fa
		List of query sequences: data/Query/MT-human_clean24.fa
	Outputs:
		1) Vector representations
			Ref genome: Output/ MT-human_clean2_genome_embeddings_sigmoid.csv
			List of query sequences: data/Query/MT-human_clean24_genome_embeddings_sigmoid.csv

		2)Perplexity: 
			Ref genome: Output/Ref/Perplexity for 2, 4, 6
			List of query sequences:  Output/Query/Perplexity for 24, 48, 72
			

Configuration 2:
	len data:  158970136 characters  (data/MT-human4.fa)
	word length = 4
        batch_size=100
	hidden_size=1000
	initial_learning rate=0.001
	num_epochs=100
	num_layers=1
	sequence size=48

	Input dataset:
		Ref genome: data/Ref/MT-human_clean6.fa
		List of query sequences:  data/Query/MT-human_clean48.fa
	Outputs:
		1) Vector representations
			Ref genome:  Output/Ref/ MT-human_clean4_genome_embeddings_sigmoid.csv
			List of query sequences:  data/Query/MT-human_clean48_genome_embeddings_sigmoid.csv
		2)Perplexity: 
			Ref genome: Output/Ref/Perplexity for 2, 4, 6
			List of query sequences:  Output/Query/Perplexity for 24, 48, 72

			

Configuration 3:
	len data:  158970136 characters  (data/Ref Genome]-MT-human6.fa)
	word length =  6
        batch_size=100
	hidden_size=1000
	initial_learning rate=0.001
	num_epochs=100
	num_layers=1
	sequence size=72

	Input dataset:
		Ref genome: data/Ref/MT-human_clean6.fa
		List of query sequences: data/Query/MT-human_clean72.fa
	Outputs:
		1) Vector representations
			Ref genome:  Output/ MT-human_clean6_genome_embeddings.csv
			List of query sequences: data/Query/MT-human_clean72_genome_embeddings_sigmoid.csv
		2)Perplexity: 
			Ref genome: Output/Ref/Perplexity for 2, 4, 6
			List of query sequences:  Output/Query/Perplexity for 24, 48, 72
			
			
Clustering Results:
Neda's Model:

|Word size | KMeans | DBScan | GMM | Number of clusters |
|--------- | ------ | ------ | --- | ------------------ |
|4 | 0.932 | 0.935 | 0.935 | 2 |
|2 | 0.184 | 0.186 | 0.186 | 2 |
|6 | 0.704 | 0.704 | 0.709 | 2 |

Lane's Model

|Word size | KMeans | DBScan | GMM | Number of clusters |
|--------- | ------ | ------ | --- | ------------------ |
| 4 | 0.305 | 0.206 | 0.305 | 2 |


