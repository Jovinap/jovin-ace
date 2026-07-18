import psutil
import os
import shutil
import platform
from typing import Dict, Any
from benchmark_tool import StorageBenchmark

def get_system_status() -> Dict[str, Any]:
    """Retrieves system hardware diagnostics and calculates flash compatibility."""
    # CPU info
    cpu_count = os.cpu_count() or 1
    cpu_percent = psutil.cpu_percent(interval=None)
    
    # RAM info
    virtual_memory = psutil.virtual_memory()
    total_ram_gb = virtual_memory.total / (1024 ** 3)
    available_ram_gb = virtual_memory.available / (1024 ** 3)
    
    # Storage info
    total_disk, used_disk, free_disk = shutil.disk_usage(".")
    total_disk_gb = total_disk / (1024 ** 3)
    free_disk_gb = free_disk / (1024 ** 3)
    
    # OS Info
    os_name = platform.system()
    os_release = platform.release()
    
    # Recommended settings based on RAM
    target_ram_cap = 4.0 # Target 4GB RAM
    fits_target = total_ram_gb >= target_ram_cap
    
    # Storage path recommendation
    storage_path = os.path.abspath(".")
    
    return {
        "os": f"{os_name} {os_release}",
        "cpu_count": cpu_count,
        "cpu_percent": cpu_percent,
        "total_ram_gb": round(total_ram_gb, 2),
        "available_ram_gb": round(available_ram_gb, 2),
        "total_disk_gb": round(total_disk_gb, 2),
        "free_disk_gb": round(free_disk_gb, 2),
        "storage_path": storage_path,
        "fits_target": fits_target,
    }

if __name__ == "__main__":
    status = get_system_status()
    for k, v in status.items():
        print(f"{k}: {v}")
