#!/usr/bin/env python3
"""Entry point for native execution of the Agent Peptide SOTA Squad inside the compute environment."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load local environment variables (.env)
load_dotenv()

def main():
    from agents_native import start_native_crew

    print("=" * 80)
    print("🚀 Launching Native DFlowNovo Research & Improvement Squad")
    print(f"🌍 GCP Project ID: {os.getenv('GCP_PROJECT_ID')}")
    print(f"📍 Vertex AI Region: {os.getenv('VERTEXAI_LOCATION', 'us-central1')}")
    print(f"📦 Workspace Root: {Path(__file__).resolve().parent}")
    print("=" * 80)
    
    result = start_native_crew(max_loops=1)
    
    print("\n" + "=" * 80)
    print("🏁 Execution Completed. Summary:")
    print("=" * 80)
    print(result)

if __name__ == "__main__":
    main()
