import typer
from rich.console import Console

from rose_cli.utils import get_client

console = Console()


def delete_thread(
    thread_id: str = typer.Argument(..., help="Thread ID to delete"),
) -> None:
    """Delete a thread and all its messages."""
    client = get_client()
    if not typer.confirm(f"Are you sure you want to delete thread {thread_id}?"):
        console.print("Cancelled.")
        return
    try:
        client.beta.threads.delete(thread_id)
        console.print(f"[red]Deleted thread: {thread_id}[/red]")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
