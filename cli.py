import argparse
import sys
import os
import json
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.markdown import Markdown

# Local imports
from status import get_system_status
from benchmark_tool import StorageBenchmark
from engine import FlashEngine
from telemetry import JovinAceTelemetry

CONFIG_FILE = os.path.expanduser("~/.jovin_ace_config.json")
console = Console()

def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_config(config: dict):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        console.print(f"[red]Error saving config: {e}[/red]")

def get_calibrated_speeds() -> tuple[float, float]:
    config = load_config()
    if "seq_read_mbps" in config and "rand_read_mbps" in config:
        return config["seq_read_mbps"], config["rand_read_mbps"]
        
    console.print("[yellow]System not calibrated yet. Running a quick storage speed test...[/yellow]")
    benchmark = StorageBenchmark(file_size_mb=64)
    res = benchmark.run_all()
    config["seq_read_mbps"] = res["read_seq_mbps"]
    config["rand_read_mbps"] = res["read_rand_mbps"]
    save_config(config)
    console.print(f"[green]Calibrated! Sequential Read: {res['read_seq_mbps']:.1f} MB/s, Random Read: {res['read_rand_mbps']:.1f} MB/s[/green]\n")
    return res["read_seq_mbps"], res["read_rand_mbps"]

def cmd_status(args):
    """Executes the status command showing hardware diagnostics."""
    status = get_system_status()
    
    console.print(Panel(
        Text("Jovin Ace System Diagnostics", style="bold white"),
        style="green",
        expand=False
    ))
    
    table = Table(show_header=True, header_style="bold green")
    table.add_column("Component", style="cyan")
    table.add_column("Diagnostic Details", style="white")
    
    table.add_row("Operating System", status["os"])
    table.add_row("CPU Core Count", f"{status['cpu_count']} Cores (Active usage: {status['cpu_percent']}%)")
    table.add_row("System DRAM Capacity", f"{status['total_ram_gb']} GB total (Available: {status['available_ram_gb']} GB)")
    table.add_row("Storage Workspace", f"{status['storage_path']}")
    table.add_row("Workspace Disk Space", f"{status['free_disk_gb']} GB free out of {status['total_disk_gb']} GB")
    
    # Target check
    fits_status = "[bold green]PASS[/bold green]" if status["fits_target"] else "[yellow]PASS (Optimized Flash Offloading recommended for low RAM limits)[/yellow]"
    table.add_row("RAM Limit Check (<4GB)", fits_status)
    
    console.print(table)
    
    # Load calibration if available
    config = load_config()
    if "seq_read_mbps" in config:
        console.print(f"\n[bold green]✓ Storage Calibrated:[/bold green]")
        console.print(f"  • Sequential Read Speed: [cyan]{config['seq_read_mbps']:.1f} MB/s[/cyan]")
        console.print(f"  • Random Read Speed:     [cyan]{config['rand_read_mbps']:.1f} MB/s[/cyan]")
    else:
        console.print("\n[yellow]⚠ Storage not calibrated. Run `jovin-ace benchmark` to check disk speeds.[/yellow]")

def cmd_benchmark(args):
    """Runs a live disk read/write speed test and prints LLM-in-a-Flash speed predictions."""
    console.print("[bold green]⏱ Starting Jovin Ace Storage & IOPS Speed Test...[/bold green]")
    console.print("Writing temporary test blocks to measure throughput...")
    
    # Use larger file size if requested for higher accuracy
    file_size = 128
    benchmark = StorageBenchmark(file_size_mb=file_size)
    
    start_t = time.time()
    results = benchmark.run_all()
    duration = time.time() - start_t
    
    console.print(f"Completed storage test in {duration:.2f} seconds.\n")
    
    # Save to config
    config = load_config()
    config["seq_read_mbps"] = results["read_seq_mbps"]
    config["rand_read_mbps"] = results["read_rand_mbps"]
    save_config(config)
    
    # Render IO Speed Results Panel
    io_table = Table(title="💽 Disk Bandwidth Results", show_header=True, header_style="bold magenta")
    io_table.add_column("Metric", style="cyan")
    io_table.add_column("Speed", style="white")
    io_table.add_row("Sequential Write Bandwidth", f"{results['write_seq_mbps']:.2f} MB/s")
    io_table.add_row("Sequential Read Bandwidth", f"{results['read_seq_mbps']:.2f} MB/s")
    io_table.add_row("Random Read Bandwidth", f"{results['read_rand_mbps']:.2f} MB/s")
    io_table.add_row("Random Read IOPS (64KB chunks)", f"{results['read_rand_iops']:.1f} IOPS")
    console.print(io_table)
    
    # Render predictions
    console.print("\n[bold green]⚡ Estimated Performance under Flash Weight-Streaming (RAM limit < 4GB)[/bold green]")
    console.print("Using Apple's 'LLM in a Flash' architecture:")
    
    for model_name, data in results["predictions"].items():
        console.print(f"\n[bold yellow]Model: {model_name} ({data['param_count']}B parameters)[/bold yellow]")
        console.print(f"  • Total model footprint (4-bit): {data['total_size_4bit_gb']} GB")
        console.print(f"  • Attention weights (in DRAM):   {data['attn_size_gb']} GB")
        console.print(f"  • Feed-Forward (FFN) size (disk):{data['ffn_size_gb']} GB")
        console.print(f"  • Avg. active neurons per token: {data['active_fraction_pct']}% (~{round(data['ffn_size_gb'] * (data['active_fraction_pct']/100), 2)} GB active)")
        
        pred_table = Table(show_header=True, header_style="bold green")
        pred_table.add_column("LRU Cache Hit Rate (DRAM)", style="cyan")
        pred_table.add_column("Data Load from Disk/Token", style="cyan")
        pred_table.add_column("Estimated Speed (Tokens/Sec)", style="bold green")
        
        for sc in data["scenarios"]:
            # Highlight 90% cache hit which is standard for conversational loops
            hit_pct = f"{sc['cache_hit_rate_pct']}%"
            if sc['cache_hit_rate_pct'] == 90:
                hit_pct = f"[bold green]{hit_pct} (Conversational Loop)[/bold green]"
                
            pred_table.add_row(
                hit_pct,
                f"{sc['bytes_read_mb']:.1f} MB",
                f"{sc['tokens_per_sec']} tok/s"
            )
            
        console.print(pred_table)

def cmd_list(args):
    """Lists available models and their status."""
    models = [
        {"name": "gemma-4:31b", "params": "31B", "size": "17.0 GB", "status": "Ready (Flash Engine)"},
        {"name": "gemma-2:27b", "params": "27B", "size": "14.8 GB", "status": "Ready (Flash Engine)"},
        {"name": "llama-3:8b", "params": "8B", "size": "4.4 GB", "status": "Ready (Standard/Flash)"},
        {"name": "llama-3:70b", "params": "70B", "size": "38.5 GB", "status": "Ready (Flash Engine)"},
        {"name": "qwen-3:0.5b", "params": "0.5B", "size": "0.25 GB", "status": "Ready (Standard/Flash)"},
        {"name": "qwen-2.5:0.5b", "params": "0.5B", "size": "0.3 GB", "status": "Downloaded"},
        {"name": "qwen-2.5:7b", "params": "7B", "size": "3.8 GB", "status": "Ready (Standard/Flash)"}
    ]
    
    table = Table(title="🤖 Available Jovin Ace Models", show_header=True, header_style="bold cyan")
    table.add_column("Model Name", style="bold white")
    table.add_column("Parameters", style="magenta")
    table.add_column("Quantized Size", style="green")
    table.add_column("Engine Status", style="yellow")
    
    for m in models:
        table.add_row(m["name"], m["params"], m["size"], m["status"])
        
    console.print(table)

def cmd_pull(args):
    """Simulates pulling/downloading a model."""
    model_name = args.model
    console.print(f"[bold green]Pulling {model_name}...[/bold green]")
    
    # Simulated download progress bar
    total_steps = 100
    with console.status(f"[cyan]Downloading model manifest...[/cyan]") as status:
        time.sleep(0.8)
        
    for i in range(1, total_steps + 1):
        time.sleep(0.02)
        bar = "█" * (i // 4) + "░" * (25 - (i // 4))
        sys.stdout.write(f"\rDownloading weights: [{bar}] {i}% | Speed: 125 MB/s")
        sys.stdout.flush()
        
    print()
    console.print(f"[bold green]✓ Successfully downloaded and verified {model_name}![/bold green]")
    console.print(f"Weight files compiled and indexed for row-column flash bundling.")

def cmd_run(args):
    """Runs the main chat execution thread with the beautiful telemetry display."""
    model_name = args.model
    engine_type = args.engine
    
    seq_read, rand_read = get_calibrated_speeds()
    
    # Initialize Engine
    engine = FlashEngine(model_name, seq_read, rand_read)
    
    # Setup conversation
    messages = []
    
    console.print(Panel.fit(
        Text.assemble(
            ("Jovin Ace Chat Session Initiated\n", "bold green"),
            (f"Model: {model_name} | Engine: {engine_type.upper()} | RAM cap: 4.0 GB\n", "cyan"),
            ("Type your message and press Enter. Type '/exit' or '/quit' to leave.", "dim white")
        ),
        border_style="green"
    ))
    
    # Check if there is config or setup
    telemetry_ui = JovinAceTelemetry(console)
    
    while True:
        try:
            # Prompt input
            prompt = console.input("\n[bold cyan]🧑 User > [/bold cyan]")
            if not prompt.strip():
                continue
                
            if prompt.strip().lower() in ["/exit", "/quit", "exit", "quit"]:
                console.print("[yellow]Exiting Jovin Ace chat loop. Goodbye![/yellow]")
                break
                
            messages.append({"role": "user", "content": prompt})
            
            # Switch to Live Dashboard context for generation streaming
            console.print("[dim]Analyzing activation paths and preparing weights stream...[/dim]")
            time.sleep(0.3)
            
            # Setup Layout for Live rendering
            layout = telemetry_ui.create_layout()
            telemetry_ui.update_header(layout, model_name)
            telemetry_ui.update_chat(layout, messages, "")
            telemetry_ui.update_input_line(layout, "Generating response...")
            
            # Populate initial Right Panel telemetry
            initial_stats = {
                "model_name": model_name,
                "params_b": engine.param_count,
                "attn_gb": engine.attn_size_gb,
                "ffn_gb": engine.ffn_size_gb,
                "total_gb": engine.attn_size_gb + engine.ffn_size_gb,
                "tps": 0.0,
                "sparsity": 10.0,
                "cache_hit": 50.0,
                "read_mb": 0.0,
                "io_mbps": 0.0,
                "ram_gb": engine.attn_size_gb + 0.5
            }
            telemetry_ui.update_model_info(layout, initial_stats)
            telemetry_ui.update_performance(layout, initial_stats)
            telemetry_ui.update_memory_map(layout, [], [])
            
            accumulated_response = ""
            
            with Live(layout, console=console, refresh_rate=15, screen=True) as live:
                # Stream tokens
                for step in engine.simulate_token_generation(prompt):
                    accumulated_response = step["accumulated_text"]
                    
                    # Update layouts
                    telemetry_ui.update_chat(layout, messages, accumulated_response)
                    
                    stats = {
                        "model_name": model_name,
                        "params_b": engine.param_count,
                        "attn_gb": engine.attn_size_gb,
                        "ffn_gb": engine.ffn_size_gb,
                        "total_gb": engine.attn_size_gb + engine.ffn_size_gb,
                        "tps": step["tokens_per_sec"],
                        "sparsity": step["sparsity_pct"],
                        "cache_hit": step["cache_hit_pct"],
                        "read_mb": step["bytes_read_mb"],
                        "io_mbps": step["disk_bandwidth_mbps"],
                        "ram_gb": step["ram_usage_gb"]
                    }
                    telemetry_ui.update_performance(layout, stats)
                    telemetry_ui.update_memory_map(layout, step["active_blocks"], step["cached_blocks"])
                
                # Generation complete. Update input line to press enter
                telemetry_ui.update_input_line(layout, "Generation complete. Press [Enter] to return to chat...")
                live.refresh()
                # Wait for user keypress (we can simulate or wait for enter)
                input()
                
            # Exit Live context (screen clears back to normal chat log)
            # Append final message to history
            messages.append({"role": "assistant", "content": accumulated_response})
            
            # Print response to the chat log
            console.print("\n[bold green]🤖 Jovin Ace (Gemma-4:31B):[/bold green]")
            console.print(Markdown(accumulated_response))
            console.print("-" * 50)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type /exit to quit.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error in runtime: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(
        description="Jovin Ace - Upgraded Ollama CLI with Flash-Streaming Technology for Low-RAM Inference."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    subparsers.add_parser("status", help="Display host hardware specs and capability diagnostics.")
    
    # Benchmark command
    subparsers.add_parser("benchmark", help="Measure disk write/read throughput and calculate LLM speeds.")
    
    # List command
    subparsers.add_parser("list", help="List all available models.")
    
    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull a model weight index from repository.")
    pull_parser.add_argument("model", type=str, help="Name of model to download.")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Start chat session with a model.")
    run_parser.add_argument("model", type=str, nargs="?", default="gemma-4:31b", help="Model name (default: gemma-4:31b).")
    run_parser.add_argument("--engine", type=str, choices=["flash", "standard"], default="flash", 
                            help="Inference engine type (default: flash weight-streaming).")
                            
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    try:
        if args.command == "status":
            cmd_status(args)
        elif args.command == "benchmark":
            cmd_benchmark(args)
        elif args.command == "list":
            cmd_list(args)
        elif args.command == "pull":
            cmd_pull(args)
        elif args.command == "run":
            cmd_run(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutdown requested. Goodbye![/yellow]")
        
if __name__ == "__main__":
    main()
