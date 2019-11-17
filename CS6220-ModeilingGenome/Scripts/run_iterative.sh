#Iterative jobs

qsub -I -q hive-gpu -l nodes=2:ppn=8:gpus=1:exclusive_process,walltime=24:00:00,mem=2gb
cd data
cd dna_align-master/
module load anaconda2
module load cuda
source activate myenv

#reference
python clean.py --path data/MT-human.fa --w 2 --savepath data/MT-human_clean
python main.py --w 2 --hidden_size 1000 --num_layers 1 --num_steps 12 --gene_name 'MT-human_clean' --save data/saved_model

python clean.py --path data/MT-human.fa --w 4 --savepath data/MT-human_clean
python main.py --w 4 --hidden_size 1000 --num_layers 1 --num_steps 12 --gene_name 'MT-human_clean' --save data/saved_model


python clean.py --path data/MT-human.fa --w 6 --savepath data/MT-human_clean
python main.py --w 6 --hidden_size 1000 --num_layers 1 --num_steps 12 --gene_name 'MT-human_clean' --save data/saved_model

#Query
python clean.py --path data/MT-human.fa --w 24 --savepath data/MT-human_clean
python main.py --w 24 --hidden_size 1000 --num_layers 1 --num_steps 1 --gene_name 'MT-human_clean' --save data/saved_model

python clean.py --path data/MT-human.fa --w 48 --savepath data/MT-human_clean
python main.py --w 48 --hidden_size 1000 --num_layers 1 --num_steps 1 --gene_name 'MT-human_clean' --save data/saved_model

python clean.py --path data/MT-human.fa --w 72 --savepath data/MT-human_clean
python main.py --w 72 --hidden_size 1000 --num_layers 1 --num_steps 1 --gene_name 'MT-human_clean' --save data/saved_model
