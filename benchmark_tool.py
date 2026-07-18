import os
import time
import random
import tempfile
import math
from typing import Dict, Any

class StorageBenchmark:
    def __init__(self, test_dir: str = None, file_size_mb: int = 128, chunk_size_kb: int = 64):
        self.test_dir = test_dir or tempfile.gettempdir()
        self.file_size_bytes = file_size_mb * 1024 * 1024
        self.chunk_size_bytes = chunk_size_kb * 1024
        self.temp_filepath = os.path.join(self.test_dir, ".jovin_ace_speedtest.tmp")

    def run_all(self) -> Dict[str, Any]:
        """Runs both sequential and random read/write tests."""
        write_seq_speed = self._test_sequential_write()
        read_seq_speed = self._test_sequential_read()
        read_rand_speed, iops = self._test_random_read()
        
        # Cleanup
        if os.path.exists(self.temp_filepath):
            try:
                os.remove(self.temp_filepath)
            except OSError:
                pass

        return {
            "write_seq_mbps": write_seq_speed,
            "read_seq_mbps": read_seq_speed,
            "read_rand_mbps": read_rand_speed,
            "read_rand_iops": iops,
            "predictions": self.calculate_predictions(read_seq_speed, read_rand_speed)
        }

    def _test_sequential_write(self) -> float:
        data = bytearray(random.getrandbits(8) for _ in range(self.chunk_size_bytes))
        chunks = self.file_size_bytes // self.chunk_size_bytes
        
        start_time = time.perf_counter()
        try:
            with open(self.temp_filepath, "wb") as f:
                for _ in range(chunks):
                    f.write(data)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            return 0.0
        
        duration = time.perf_counter() - start_time
        speed_mbps = (self.file_size_bytes / (1024 * 1024)) / duration
        return speed_mbps

    def _test_sequential_read(self) -> float:
        if not os.path.exists(self.temp_filepath):
            return 0.0
            
        start_time = time.perf_counter()
        total_read = 0
        try:
            with open(self.temp_filepath, "rb") as f:
                while True:
                    chunk = f.read(self.chunk_size_bytes)
                    if not chunk:
                        break
                    total_read += len(chunk)
        except Exception as e:
            return 0.0
            
        duration = time.perf_counter() - start_time
        speed_mbps = (total_read / (1024 * 1024)) / duration
        return speed_mbps

    def _test_random_read(self) -> tuple[float, float]:
        if not os.path.exists(self.temp_filepath):
            return 0.0, 0.0
            
        num_reads = 1000  # Number of random lookups
        max_seek = self.file_size_bytes - self.chunk_size_bytes
        if max_seek <= 0:
            return 0.0, 0.0
            
        seeks = [random.randint(0, max_seek) for _ in range(num_reads)]
        
        start_time = time.perf_counter()
        total_read = 0
        try:
            with open(self.temp_filepath, "rb") as f:
                for pos in seeks:
                    f.seek(pos)
                    chunk = f.read(self.chunk_size_bytes)
                    total_read += len(chunk)
        except Exception as e:
            return 0.0, 0.0
            
        duration = time.perf_counter() - start_time
        speed_mbps = (total_read / (1024 * 1024)) / duration
        iops = num_reads / duration
        return speed_mbps, iops

    def calculate_predictions(self, seq_read_mbps: float, rand_read_mbps: float) -> Dict[str, Any]:
        """Calculates token generation speeds under different configurations."""
        # Models to evaluate
        # Format: (name, param_count_billion, default_active_fraction)
        models = [
            ("Qwen-3-0.5B", 0.5, 0.20),
            ("Llama-3-8B", 8.0, 0.15),
            ("Gemma-4-31B", 31.0, 0.10),
            ("Llama-3-70B", 70.0, 0.08)
        ]
        
        results = {}
        for name, params, active_fraction in models:
            # Model sizes in GB for different quantization levels: 4-bit, 3-bit, 2-bit
            # 4-bit quantization = ~0.5 bytes per parameter + some overhead
            size_4bit_gb = params * 0.55
            
            # FFN parameters make up roughly 2/3 of the model
            ffn_size_gb = size_4bit_gb * (2/3)
            # Attention layers make up 1/3 (kept in RAM, say ~2GB for 31B, which fits in 4GB RAM)
            attn_size_gb = size_4bit_gb * (1/3)
            
            # With windowing and neuron caching, let's assume a Cache Hit Rate (LRU)
            # The cache hit rate for FFN weights can range from 50% to 90% depending on text coherence
            # We show three scenarios: 
            # 1. No cache (raw streaming from flash)
            # 2. Balanced cache (60% hit rate)
            # 3. High cache (90% hit rate)
            
            scenarios = []
            for cache_hit_rate in [0.0, 0.60, 0.90]:
                # Effective FFN bytes loaded per token from disk:
                # FFN_size * active_fraction * (1 - cache_hit_rate)
                active_ffn_gb = ffn_size_gb * active_fraction
                bytes_loaded_gb = active_ffn_gb * (1 - cache_hit_rate)
                
                # Convert GB to MB
                bytes_loaded_mb = bytes_loaded_gb * 1024
                
                # Assume effective read speed is a blend of sequential and random read (70% seq, 30% rand)
                effective_read_speed = (seq_read_mbps * 0.7) + (rand_read_mbps * 0.3)
                if effective_read_speed <= 0:
                    effective_read_speed = 10.0 # safety fallback
                
                # Tokens per second = read_speed / bytes_loaded
                tokens_per_sec = effective_read_speed / bytes_loaded_mb
                
                # Limit tokens per sec to CPU/GPU processing limits (e.g. max 80 tokens/sec)
                tokens_per_sec = min(tokens_per_sec, 80.0)
                
                scenarios.append({
                    "cache_hit_rate_pct": int(cache_hit_rate * 100),
                    "bytes_read_mb": round(bytes_loaded_mb, 2),
                    "tokens_per_sec": round(tokens_per_sec, 1)
                })
                
            results[name] = {
                "param_count": params,
                "total_size_4bit_gb": round(size_4bit_gb, 2),
                "ffn_size_gb": round(ffn_size_gb, 2),
                "attn_size_gb": round(attn_size_gb, 2),
                "active_fraction_pct": int(active_fraction * 100),
                "scenarios": scenarios
            }
            
        return results

if __name__ == "__main__":
    import sys
    print("Running speed test...")
    benchmark = StorageBenchmark(file_size_mb=64)
    res = benchmark.run_all()
    print(f"Seq Read: {res['read_seq_mbps']:.2f} MB/s")
    print(f"Rand Read: {res['read_rand_mbps']:.2f} MB/s (IOPS: {res['read_rand_iops']:.1f})")
    print("\nPredictions for Gemma-4-31B (4-bit, size: 17.0 GB):")
    gemma_pred = res["predictions"]["Gemma-4-31B"]
    for sc in gemma_pred["scenarios"]:
        print(f"  Cache Hit {sc['cache_hit_rate_pct']}%: Read {sc['bytes_read_mb']:.1f} MB/token -> {sc['tokens_per_sec']} tok/s")
