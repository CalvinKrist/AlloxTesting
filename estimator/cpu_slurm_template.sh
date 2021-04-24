#!/bin/sh
#SBATCH --partition=main
#SBATCH --time=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="<NAME>"
#SBATCH --exclusive
#SBATCH --output=<OUTPUT>
#SBATCH --requeue
#SBATCH --threads-per-core=<THREADS>
