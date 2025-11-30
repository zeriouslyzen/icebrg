"""
HPC Integration
High-Performance Computing cluster integration
"""

from typing import Any, Dict, Optional, List
import subprocess
import os


class HPCIntegration:
    """HPC cluster integration for supercomputer-grade simulations"""
    
    def __init__(self):
        self.mpi_available = False
        self.openmp_available = False
        self.slurm_available = False
        
        # Check for MPI
        try:
            result = subprocess.run(['mpirun', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.mpi_available = True
        except Exception:
            pass
        
        # Check for Slurm
        try:
            result = subprocess.run(['squeue', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.slurm_available = True
        except Exception:
            pass
    
    def submit_job(
        self,
        script: str,
        nodes: int = 1,
        tasks_per_node: int = 1,
        time_limit: str = "01:00:00",
        memory: str = "4GB"
    ) -> Dict[str, Any]:
        """Submit HPC job"""
        if self.slurm_available:
            return self._submit_slurm_job(script, nodes, tasks_per_node, time_limit, memory)
        else:
            return self._submit_local_job(script)
    
    def _submit_slurm_job(
        self,
        script: str,
        nodes: int,
        tasks_per_node: int,
        time_limit: str,
        memory: str
    ) -> Dict[str, Any]:
        """Submit job to Slurm scheduler"""
        try:
            # Create Slurm script
            slurm_script = f"""#!/bin/bash
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --time={time_limit}
#SBATCH --mem={memory}
#SBATCH --job-name=iceburg_job

{script}
"""
            
            # Write script to file
            script_path = "/tmp/iceburg_slurm_job.sh"
            with open(script_path, 'w') as f:
                f.write(slurm_script)
            
            # Submit job
            result = subprocess.run(
                ['sbatch', script_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                return {
                    "success": True,
                    "job_id": job_id,
                    "scheduler": "slurm"
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _submit_local_job(self, script: str) -> Dict[str, Any]:
        """Submit job locally"""
        try:
            result = subprocess.run(
                script,
                shell=True,
                capture_output=True,
                text=True
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "scheduler": "local"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_parallel(
        self,
        tasks: List[callable],
        processes: int = 4
    ) -> List[Any]:
        """Run tasks in parallel using MPI or multiprocessing"""
        if self.mpi_available and processes > 1:
            return self._run_mpi(tasks, processes)
        else:
            return self._run_multiprocessing(tasks, processes)
    
    def _run_mpi(self, tasks: List[callable], processes: int) -> List[Any]:
        """Run tasks using MPI"""
        # Placeholder for MPI execution
        # In production, would use mpi4py
        return self._run_multiprocessing(tasks, processes)
    
    def _run_multiprocessing(self, tasks: List[callable], processes: int) -> List[Any]:
        """Run tasks using multiprocessing"""
        from multiprocessing import Pool
        
        try:
            with Pool(processes=processes) as pool:
                results = pool.map(lambda f: f(), tasks)
            return results
        except Exception as e:
            return []
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        if self.slurm_available:
            try:
                result = subprocess.run(
                    ['squeue', '-j', job_id, '--format=%T,%M,%N'],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        status = lines[1].split(',')
                        return {
                            "job_id": job_id,
                            "status": status[0] if len(status) > 0 else "UNKNOWN",
                            "time": status[1] if len(status) > 1 else "",
                            "nodes": status[2] if len(status) > 2 else ""
                        }
            except Exception:
                pass
        
        return {
            "job_id": job_id,
            "status": "UNKNOWN"
        }
    
    def get_available_resources(self) -> Dict[str, Any]:
        """Get available HPC resources"""
        resources = {
            "mpi_available": self.mpi_available,
            "slurm_available": self.slurm_available,
            "openmp_available": self.openmp_available
        }
        
        if self.slurm_available:
            try:
                result = subprocess.run(
                    ['sinfo', '--format=%P,%l,%D,%T'],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    resources["cluster_info"] = result.stdout
            except Exception:
                pass
        
        return resources

