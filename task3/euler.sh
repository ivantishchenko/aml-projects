bsub -W 5 -n 4 -R rusage[mem=4096] "module load python/3.6.0 && python3 gen_data_cluster.py"

bsub -W 3:59 -n 4 -R rusage[mem=16384] "module load python/3.6.0 && python3 task3_ivan.py COMBO"

bsub -W 4:00 -n2 -R rusage[mem=8192,ngpus_excl_p=1] "module load gcc/6.3.0 python_gpu/3.6.4 && python3 nn_cil.py"