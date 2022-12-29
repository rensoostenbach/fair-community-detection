from simple_slurm import Slurm

slurm = Slurm(
    nodes=1,
    ntasks=8,
    job_name="fair-community-detection",
    partition="mcs.default.q",
    error=f"{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.err",
    output=f"{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out",
)
slurm.sbatch("python comparison_cd_algs.py")
