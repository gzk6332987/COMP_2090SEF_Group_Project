from rich import print as rprint
import torch

def check_cuda_status():
    """
    check computer cuda environment
    """
    rprint("[bold cyan]Checking your cuda environment ...[/bold cyan]")
    
    cuda_available = torch.cuda.is_available()
    
    if not cuda_available:
        rprint("[bold red]❌ cuda is not enable in your device[/bold red] We will use CPU perform inference and training")
        rprint("[yellow]You can search online learn how to enable[/yellow]")
        return torch.device("cpu")

    # 2. 獲取設備信息
    device_id = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_id)
    cuda_version = torch.version.cuda
    
    # 3. 獲取顯存狀態 (以 GB 為單位)
    # 使用 torch.cuda.mem_get_info() 獲取實時數據 [PyTorch Docs]
    free_mem, total_mem = torch.cuda.mem_get_info(device_id)
    free_gb = free_mem / 1024**3
    total_gb = total_mem / 1024**3

    rprint(f"[green]✅ CUDA Is useable in your computer![/green]")
    rprint(f"   - device: [magenta]{device_name}[/magenta]")
    rprint(f"   - CUDA version: [yellow]{cuda_version}[/yellow]")
    rprint(f"   - CUDA memory: [blue]{free_gb:.2f}GB[/blue] / {total_gb:.2f}GB (useable/total)")

    # 4. 針對 Python 3.14 的特殊警告
    if torch.cuda.get_device_capability(device_id)[0] < 7:
        rprint("[yellow]⚠️ Warning: Your GPU has limited computing power; training an 800-word LSTM may be slow.[/yellow]")