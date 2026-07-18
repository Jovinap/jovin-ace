from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console
from rich.progress import ProgressBar
from typing import Dict, Any, List

class JovinAceTelemetry:
    def __init__(self, console: Console):
        self.console = console
        self.num_blocks = 64
        
    def create_layout(self) -> Layout:
        """Creates a split screen terminal layout."""
        layout = Layout()
        
        # Split into main content (top/left) and sidebar/telemetry (right)
        layout.split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=2)
        )
        
        # Split left panel into Chat History and status line
        layout["left"].split_column(
            Layout(name="header", size=3),
            Layout(name="chat", ratio=9),
            Layout(name="input_line", size=3)
        )
        
        # Split right panel into multiple telemetry cards
        layout["right"].split_column(
            Layout(name="model_info", size=7),
            Layout(name="performance", size=9),
            Layout(name="memory_map", ratio=1)
        )
        
        return layout

    def update_header(self, layout: Layout, model_name: str):
        """Updates top header panel."""
        header_text = Text.assemble(
            (" JOVIN ACE v1.0.0 ", "bold white on green"),
            (" ─── Running high-Billion LLMs on Low RAM using Flash-Streaming ───", "dim white")
        )
        layout["left"]["header"].update(Panel(header_text, style="green"))

    def update_chat(self, layout: Layout, messages: List[Dict[str, str]], current_streaming: str = ""):
        """Renders chat history and active streaming tokens."""
        chat_text = Text()
        
        # Render past messages
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                chat_text.append("\n🧑 User: ", style="bold cyan")
                chat_text.append(content + "\n")
            elif role == "assistant":
                chat_text.append("\n🤖 Jovin Ace (Gemma-4:31B): ", style="bold green")
                chat_text.append(content + "\n")
                
        # Render current streaming token if any
        if current_streaming:
            chat_text.append("\n🤖 Jovin Ace (Gemma-4:31B): ", style="bold green")
            chat_text.append(current_streaming)
            # Append a cursor block
            chat_text.append("█", style="blink green")
            
        layout["left"]["chat"].update(Panel(chat_text, title="Interactive Chat Console", border_style="cyan"))

    def update_input_line(self, layout: Layout, prompt_text: str = ""):
        """Renders user input line indicator."""
        input_text = Text.assemble(
            (" Prompt > ", "bold cyan"),
            (prompt_text, "white")
        )
        layout["left"]["input_line"].update(Panel(input_text, border_style="dim cyan"))

    def update_model_info(self, layout: Layout, stats: Dict[str, Any]):
        """Renders model loading stats and architecture specifications."""
        table = Table(show_header=False, expand=True, box=None)
        table.add_row(Text("Model Name:", style="bold yellow"), Text(stats.get("model_name", "Gemma-4-31B"), style="yellow"))
        table.add_row(Text("Parameter Count:", style="dim white"), Text(f"{stats.get('params_b', 31.0)} Billion", style="white"))
        table.add_row(Text("Attention in RAM:", style="dim white"), Text(f"{stats.get('attn_gb', 5.2)} GB", style="green"))
        table.add_row(Text("FFN on Storage:", style="dim white"), Text(f"{stats.get('ffn_gb', 10.4)} GB", style="magenta"))
        table.add_row(Text("Total Size (4-bit):", style="dim white"), Text(f"{stats.get('total_gb', 15.6)} GB", style="white"))
        
        layout["right"]["model_info"].update(Panel(table, title="⚙️ Model Architecture", border_style="yellow"))

    def update_performance(self, layout: Layout, stats: Dict[str, Any]):
        """Renders live performance dials and read statistics."""
        table = Table(show_header=False, expand=True, box=None)
        
        # Generation Speed
        tps = stats.get("tps", 0.0)
        tps_style = "bold green" if tps >= 45.0 else ("green" if tps >= 20.0 else "red")
        table.add_row(Text("Generation Rate:", style="bold white"), Text(f"{tps} tok/sec", style=tps_style))
        
        # Dynamic Sparsity
        sparsity = stats.get("sparsity", 10.0)
        table.add_row(Text("Neuron Sparsity:", style="dim white"), Text(f"{sparsity}% FFN active", style="magenta"))
        
        # LRU Cache Hit Rate
        hit_rate = stats.get("cache_hit", 0.0)
        hit_style = "bold green" if hit_rate >= 90.0 else ("yellow" if hit_rate >= 60.0 else "red")
        table.add_row(Text("Cache Hit Rate:", style="dim white"), Text(f"{hit_rate}%", style=hit_style))
        
        # Storage Read
        read_mb = stats.get("read_mb", 0.0)
        table.add_row(Text("Data Read/Token:", style="dim white"), Text(f"{read_mb} MB", style="cyan"))
        
        # Disk IO Bandwidth
        io_speed = stats.get("io_mbps", 0.0)
        table.add_row(Text("Disk IO Throughput:", style="dim white"), Text(f"{io_speed:.1f} MB/s", style="cyan"))
        
        # Active DRAM
        ram_used = stats.get("ram_gb", 0.0)
        max_ram = 4.0
        table.add_row(
            Text("DRAM Usage:", style="dim white"), 
            Text(f"{ram_used:.2f} GB / {max_ram} GB max", style="bold green" if ram_used <= max_ram else "bold red")
        )
        
        layout["right"]["performance"].update(Panel(table, title="⚡ Flash Streaming Telemetry", border_style="green"))

    def update_memory_map(self, layout: Layout, active_blocks: List[int], cached_blocks: List[int]):
        """Draws a visual 8x8 grid of weight blocks mapping storage to memory."""
        grid_text = Text()
        grid_text.append("Storage Blocks Mapping (64 Blocks x 160MB)\n\n", style="dim white")
        
        for i in range(self.num_blocks):
            if i in active_blocks:
                # Active neuron weights being streamed from SSD right now (yellow triangle)
                grid_text.append("▲ ", style="bold yellow")
            elif i in cached_blocks:
                # Cached neuron weights residing in DRAM (green square)
                grid_text.append("■ ", style="bold green")
            else:
                # Inactive neuron weights residing on Disk/Storage (grey dot)
                grid_text.append("· ", style="dim white")
                
            if (i + 1) % 8 == 0:
                grid_text.append("\n")
                
        grid_text.append("\nLegend: ", style="dim white")
        grid_text.append("·", style="dim white")
        grid_text.append(" Storage  ", style="white")
        grid_text.append("■", style="bold green")
        grid_text.append(" RAM Cache  ", style="white")
        grid_text.append("▲", style="bold yellow")
        grid_text.append(" I/O Stream", style="white")
        
        layout["right"]["memory_map"].update(Panel(grid_text, title="📂 Memory Map Grid", border_style="magenta"))
