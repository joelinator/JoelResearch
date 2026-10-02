#!/usr/bin/env python3
"""Entry point for Cloud Run Job execution of the Agent Peptide SOTA Squad."""
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def sync_remote_research_branch():
    """Guarantee that the orchestrator workspace strictly uses the remote 'research' branch code."""
    repo_url = os.getenv("GITHUB_REPO_URL", "https://github.com/joelinator/joelresearch.git")
    print("=" * 80)
    print(f"📥 [SYNCHRONIZATION] Fetching remote 'research' branch from {repo_url}...")
    try:
        if (Path.cwd() / ".git").exists():
            subprocess.run(["git", "remote", "set-url", "origin", repo_url], check=False)
            subprocess.run(["git", "fetch", "origin", "research"], check=True)
            subprocess.run(["git", "checkout", "-B", "research", "origin/research"], check=True)
            subprocess.run(["git", "reset", "--hard", "origin/research"], check=True)
            head_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            print(f"✅ Successfully synchronized workspace to remote origin/research (Commit: {head_commit[:8]}).")
        else:
            print("ℹ️ Workspace is not a git repo, cloning remote 'research' branch...")
            subprocess.run(["git", "clone", "--branch", "research", repo_url, "/tmp/remote_research"], check=True)
            print("✅ Cloned fresh remote 'research' branch.")
    except Exception as e:
        print(f"⚠️ Remote sync notice: {e}")
    print("=" * 80)


def main():
    sync_remote_research_branch()
    
    from agents import start_collaborative_crew, stop_vm_safely

    print("=" * 80)
    print("🚀 Initializing Optimized Multi-Agent R&D Squad for Peptide Sequencing")
    print(f"🌍 GCP Project ID: {os.getenv('GCP_PROJECT_ID')}")
    print(f"⚡ Target A100 VM: {os.getenv('A100_VM_NAME')} (Zone: {os.getenv('A100_VM_ZONE')})")
    print(f"📦 Remote Repo: {os.getenv('GITHUB_REPO_URL')} (Branch: research)")
    print("=" * 80)
    
    try:
        result = start_collaborative_crew()
        print("\n" + "=" * 80)
        print("🏁 Execution Completed. Results Summary:")
        print("=" * 80)
        print(result)
    finally:
        stop_vm_safely()

if __name__ == "__main__":
    main()
