import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def csv_to_jsonl(csv_path: str, output_path: str, schema: str = "fault"):
    """
    Convert fault_data.csv to JSONL format.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output JSONL file
        schema: Either "fault" or "instruction"
    """
    df = pd.read_csv(csv_path)
    
    with open(output_path, 'w') as f:
        for idx, row in df.iterrows():
            if schema == "fault":
                # Fault schema format
                record = {
                    "id": row['Fault ID'],
                    "timestamp": datetime.now().isoformat() + "Z",
                    "source": "historical_data",
                    "text": f"Fault {row['Fault ID']}: {row['Fault Type']} at location {row['Fault Location (Latitude, Longitude)']}. "
                            f"Voltage: {row['Voltage (V)']}V, Current: {row['Current (A)']}A, Power Load: {row['Power Load (MW)']}MW. "
                            f"Temperature: {row['Temperature (°C)']}°C, Wind Speed: {row['Wind Speed (km/h)']}km/h, "
                            f"Weather: {row['Weather Condition']}, Maintenance: {row['Maintenance Status']}, "
                            f"Component Health: {row['Component Health']}. "
                            f"Fault Duration: {row['Duration of Fault (hrs)']}hrs, Downtime: {row['Down time (hrs)']}hrs.",
                    "fault_type": row['Fault Type'],
                    "labels": [row['Weather Condition'], row['Component Health'], row['Maintenance Status']],
                    "recommendations": None,
                    "metadata": {
                        "location": row['Fault Location (Latitude, Longitude)'],
                        "voltage": row['Voltage (V)'],
                        "current": row['Current (A)'],
                        "power_load": row['Power Load (MW)'],
                        "temperature": row['Temperature (°C)'],
                        "wind_speed": row['Wind Speed (km/h)'],
                        "duration_hrs": row['Duration of Fault (hrs)'],
                        "downtime_hrs": row['Down time (hrs)']
                    }
                }
            else:  # instruction schema
                # Create instruction-style format
                record = {
                    "id": row['Fault ID'],
                    "timestamp": datetime.now().isoformat() + "Z",
                    "source": "historical_data",
                    "instruction": "Classify the fault type and suggest actions based on the power system data.",
                    "input": f"Location: {row['Fault Location (Latitude, Longitude)']}. "
                            f"Voltage: {row['Voltage (V)']}V, Current: {row['Current (A)']}A, Power Load: {row['Power Load (MW)']}MW. "
                            f"Temperature: {row['Temperature (°C)']}°C, Wind Speed: {row['Wind Speed (km/h)']}km/h, "
                            f"Weather: {row['Weather Condition']}, Maintenance: {row['Maintenance Status']}, "
                            f"Component Health: {row['Component Health']}. "
                            f"Fault Duration: {row['Duration of Fault (hrs)']}hrs, Downtime: {row['Down time (hrs)']}hrs.",
                    "output": f"Fault Type: {row['Fault Type']}. "
                             f"Action: Inspect {row['Component Health'].lower()} component. "
                             f"Priority: {'High' if row['Down time (hrs)'] > 5 else 'Medium' if row['Down time (hrs)'] > 3 else 'Low'}. "
                             f"Weather consideration: {row['Weather Condition']} conditions may have contributed.",
                    "tags": [row['Fault Type'], row['Weather Condition'], row['Component Health']],
                    "metadata": {
                        "location": row['Fault Location (Latitude, Longitude)'],
                        "downtime_hrs": row['Down time (hrs)']
                    }
                }
            
            f.write(json.dumps(record) + '\n')
    
    print(f"✓ Converted {len(df)} records from {csv_path} to {output_path}")

if __name__ == "__main__":
    import typer
    app = typer.Typer()
    
    @app.command()
    def main(
        csv_path: str = typer.Option("data/samples/fault_data.csv", help="Input CSV file"),
        output_path: str = typer.Option("data/processed/fault_data.jsonl", help="Output JSONL file"),
        schema: str = typer.Option("instruction", help="Schema type: 'fault' or 'instruction'")
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        csv_to_jsonl(csv_path, output_path, schema)
    
    app()