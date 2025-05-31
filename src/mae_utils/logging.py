# logging.py
import subprocess
import psutil
import time

def get_system_metrics(start_time=time.time()):
    metrics = {}

    # Time
    metrics['utils/timestamp'] = time.time()
    metrics['utils/rel_timestamp'] = time.time() - start_time

    # CPU utilization
    metrics["utils/cpu_util"] = psutil.cpu_percent()

    # RAM utilization
    metrics["utils/ram_percent"] = psutil.virtual_memory().percent

    # Swap utilization
    metrics["utils/swap_percent"] = psutil.swap_memory().percent

    # CPU memory usage
    process = psutil.Process()
    mem_info = process.memory_info()
    metrics["utils/cpu_rss_mb"] = mem_info.rss / (1024 * 1024)

    # GPU utilization, memory utilization and temperature, using nvidia-smi
    try:
        gpu_query = 'utilization.gpu,utilization.memory,temperature.gpu'
        res = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=' + gpu_query,
             '--format=csv,nounits,noheader']
        )
        lines = [line for line in res.decode().split('\n') if line != '']
        for i_gpu, line in enumerate(lines):
            gpu_util, mem_util, temp = line.split(',')
            metrics[f"utils/gpu_{i_gpu}_util"] = float(gpu_util.strip()) if 'Not Supported' not in gpu_util else None
            metrics[f"utils/gpu_{i_gpu}_mem_percent"] = float(mem_util.strip()) if 'Not Supported' not in mem_util else None
            metrics[f"utils/gpu_{i_gpu}_temp"] = float(temp.strip()) if 'Not Supported' not in temp else None

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error querying nvidia-smi: {e}", file=sys.stderr)
        # If nvidia-smi fails, don't add any GPU metrics.

    return metrics