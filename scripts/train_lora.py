import typer
import sys
from pathlib import Path

# Ensure 'src' is on the path when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fault_analysis.training.lora_train import train

app = typer.Typer()


@app.command()
def main(config: str = typer.Option(..., help="Path to YAML config")):
    train(config)


if __name__ == "__main__":
    app()
