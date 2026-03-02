from rich import print as rprint
import torch

def check_cuda_status():
    """
    check computer cuda environment
    """
    rprint("[bold cyan]Checking your CUDA environment...[/bold cyan]")
    
    is_cuda_available = torch.cuda.is_available()
    
    if not is_cuda_available:
        rprint("[bold red]❌ cuda is not enable in your device[/bold red] We will use CPU perform inference and training")
        rprint("[yellow]If your exactly have NVIDIA GPU, you can search online learn how to enable :>[/yellow]")

    # get device info
    device_id = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_id)
    cuda_version = torch.version.cuda
    
    # get memory state (Unit: GiB)
    free_mem, total_mem = torch.cuda.mem_get_info(device_id)
    free_gb = free_mem / 1024**3
    total_gb = total_mem / 1024**3

    rprint(f"[green]✅ CUDA Is useable in your computer![/green]")
    rprint(f"   - device: [magenta]{device_name}[/magenta]")
    rprint(f"   - CUDA version: [yellow]{cuda_version}[/yellow]")
    rprint(f"   - CUDA memory: [blue]{free_gb:.2f}GB[/blue] / {total_gb:.2f}GB (useable/total)")

    # warning for low memory user
    if torch.cuda.get_device_capability(device_id)[0] < 7:
        rprint("[yellow]⚠️ Warning: Your GPU has limited computing power; training and inferring an LSTM may be slow.[/yellow]")