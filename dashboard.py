import os
import subprocess

def launch_dashboard():
    print("Launching MLflow UI...")
    print("Dashboard will be available at http://localhost:5000")
    try:
        # Run mlflow ui in the background
        subprocess.run(["uv", "run", "mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"])
    except KeyboardInterrupt:
        print("\nStopping dashboard.")

if __name__ == "__main__":
    launch_dashboard()
