#!/usr/bin/env python3
"""
DFlowNovo Multi-Agent Autonomous Research & Improvement Squad (Native Job Runner)
=================================================================================
Automated launcher for native execution inside the compute environment.
Automatically configures GitHub CLI (gh) credentials, detects live GCP / Vertex AI
authentication, displays system health diagnostics, and launches the collaborative squad.
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load local environment variables (.env)
load_dotenv()

WORKSPACE_ROOT = Path(__file__).resolve().parent

def setup_github_authentication():
    """Configures GitHub git credentials using the active gh CLI login."""
    print("🔑 Configuring GitHub credentials...")
    try:
        # Check gh auth status
        status_res = subprocess.run(
            ["gh", "auth", "status"], capture_output=True, text=True, cwd=str(WORKSPACE_ROOT)
        )
        if status_res.returncode == 0:
            print("  ✓ gh CLI is logged in.")
            # Set up git credential helper for gh
            subprocess.run(["gh", "auth", "setup-git"], check=False, cwd=str(WORKSPACE_ROOT))
            print("  ✓ Configured git credential helper via 'gh auth setup-git'.")
            
            # Extract token if GITHUB_TOKEN is not already set
            if not os.getenv("GITHUB_TOKEN"):
                token = subprocess.check_output(
                    ["gh", "auth", "token"], text=True, stderr=subprocess.DEVNULL, cwd=str(WORKSPACE_ROOT)
                ).strip()
                if token:
                    os.environ["GITHUB_TOKEN"] = token
                    print("  ✓ Exported GITHUB_TOKEN from gh CLI session.")
        else:
            print("  ⚠️ gh CLI not logged in or status returned non-zero.")
    except Exception as e:
        print(f"  ⚠️ Notice: Could not configure gh credentials automatically: {e}")


def display_environment_dashboard():
    """Prints a comprehensive diagnostic dashboard of system resources, environment, and auth configuration."""
    project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GCP_PROJECT", "joelgedeon-project-507410")
    region = os.getenv("VERTEXAI_LOCATION", "us-central1")
    model = os.getenv("CREWAI_MODEL", "gemini/gemini-2.5-pro")
    
    # Check GPU
    gpu_info = "Not Available (CPU mode)"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            gpu_info = f"✓ {gpu_name} ({vram_gb} GB VRAM)"
    except Exception:
        pass

    # Check OAuth token file
    token_file = Path.home() / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
    auth_mode = "Vertex AI ADC / Project Default"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        auth_mode = "Google AI Studio API Key"
    elif token_file.exists():
        auth_mode = "Antigravity Live OAuth Token (Cloud Platform Scope)"

    print("=" * 80)
    print("🚀 DFlowNovo Autonomous Research, Improvement & Verification Squad")
    print("=" * 80)
    print(f"  🐍 Python Interpreter : {sys.executable}")
    print(f"  📦 Workspace Root      : {WORKSPACE_ROOT}")
    print(f"  ⚡ Compute Device      : {gpu_info}")
    print(f"  🌍 GCP Project ID      : {project_id}")
    print(f"  📍 Vertex AI Region    : {region}")
    print(f"  🤖 LLM Engine          : {model}")
    print(f"  🔐 LLM Authentication  : {auth_mode}")
    has_git_auth = bool(os.getenv("GITHUB_TOKEN") or shutil.which("gh"))
    print(f"  🐙 GitHub Push Auth    : {'✓ Active (gh CLI)' if has_git_auth else '⚠️ Not configured'}")
    print("=" * 80)


def main():
    start_time = time.time()
    
    # 1. Setup GitHub credentials via gh CLI
    setup_github_authentication()
    
    # 2. Display environment dashboard
    display_environment_dashboard()
    
    # 3. Launch the native multi-agent crew
    from agents_native import start_native_crew
    print("\n🏁 Launching Native CrewAI Research Execution Loop...")
    result = start_native_crew(max_loops=1)
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✅ Research Squad Run Completed in {elapsed:.1f}s. Summary Output:")
    print("=" * 80)
    print(result)


if __name__ == "__main__":
    main()
