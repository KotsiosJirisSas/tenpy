#!/bin/bash -l
echo =========================================================   
echo Job submitted  date = Thu  8 Oct 15:18:36 BST 2026      
date_start=`date +%s`
echo $SLURM_JOB_NUM_NODES nodes \( $SLURM_CPUS_ON_NODE processes per node \)        
echo $SLURM_JOB_NUM_NODES hosts used: $SLURM_JOB_NODELIST      
# Set this otherwise a different transport gets selected on some nodes and things break in strange ways
export OMPI_MCA_pml=^cm
echo Job output begins                                           
echo -----------------                                           
echo   
#hostname

# Need to set the max locked memory very high otherwise IB can't allocate enough and fails with "UCX  ERROR Failed to allocate memory pool chunk: Input/output error"
ulimit -l unlimited

export OMP_NUM_THEADS=1
 /usr/local/shared/slurm/bin/srun -u -n 1 --mpi=pmix --mem-per-cpu=10240 nice -n 10 /usr/bin/python3 /mnt/users/dperkovic/DMRG_modified/tenpy/segment_DMRG_final/multilayer_segment_final.py 0 /mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/nu=2_3_antidot_project/0LL_d=6_Lx=15.0_coulumb_double_gated_pstate=[1,0,1]_chi=400_MPS_data_new_tenpy govno 60 20 1000 1 0 0
  echo ---------------                                           
  echo Job output ends                                           

  date_end=`date +%s`
  seconds=$((date_end-date_start))
  minutes=$((seconds/60))
  seconds=$((seconds-60*minutes))
  hours=$((minutes/60))
  minutes=$((minutes-60*hours))
  echo =========================================================   
  echo PBS job: finished   date = `date`   
  echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
  echo =========================================================
