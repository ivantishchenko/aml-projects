bsub -W 5 -n 4 -R rusage[mem=4096] "module load python/3.6.0 && python3 gen_data_cluster.py"

bsub -W 3:59 -n 4 -R rusage[mem=16384] "module load python/3.6.0 && python3 task3_ivan.py COMBO"