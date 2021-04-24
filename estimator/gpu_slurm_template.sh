#!/bin/sh
#SBATCH --partition=main
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="<NAME>"
#SBATCH --exclusive
#SBATCH --output=<OUTPUT>
#SBATCH --requeue