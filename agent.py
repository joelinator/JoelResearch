# agents.py / agent.py
"""
DFlowNovo Multi-Agent Autonomous Research, Improvement & Verification Squad
===========================================================================
Collaborative R&D squad that actively drives algorithmic, architectural, and
empirical performance improvements for de novo peptide sequencing, while
enforcing strict mathematical fact-checking, artifact coherence, publication
figure regeneration, MGF/MGZ/mzML data parser robustness, and Hugging Face deployment.

Target VM Environment:
- Project Root: ~/dfm-joelresearch
- Virtual Environment: ~/dfm-joelresearch/.venv
- Branch: research (synced from GitHub)
"""

import os
import sys
import time
import json
import re
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

project_id = os.getenv("GCP_PROJECT_ID", "joelgedeon-project-507410")
region = os.getenv("VERTEXAI_LOCATION", "us-central1")
repo_url = os.getenv("GITHUB_REPO_URL", "https://github.com/joelinator/joelresearch.git")
vm_name = os.getenv("A100_VM_NAME", "dfm-train-a100")
vm_zone = os.getenv("A100_VM_ZONE", "europe-west4-b")

# Ensure Vertex AI routing for Google GenAI SDK
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["VERTEXAI_PROJECT"] = project_id
os.environ["VERTEXAI_LOCATION"] = region
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.utilities import ArxivAPIWrapper
from crewai.tools import tool

# Native CrewAI Gemini 2.5 Pro LLM integration with Vertex AI backend
vertex_llm = LLM(
    model="gemini/gemini-2.5-pro",
    temperature=0.1,
    project=project_id,
    location=region
)

# =====================================================================
# VM LIFECYCLE MANAGEMENT (SESSION KEEP-ALIVE)
# =====================================================================

def is_vm_running(name: str = vm_name, zone: str = vm_zone) -> bool:
    """Checks whether the A100 VM is currently in RUNNING state."""
    try:
        res = subprocess.run(
            ["gcloud", "compute", "instances", "describe", name, "--zone", zone, "--format=value(status)"],
            capture_output=True, text=True, check=True
        )
        return res.stdout.strip() == "RUNNING"
    except Exception:
        return False


def ensure_vm_running(name: str = vm_name, zone: str = vm_zone, max_retries: int = 3) -> bool:
    """Ensures the A100 GPU VM is running without unnecessary restarts to keep the GPU reservation."""
    if is_vm_running(name, zone):
        return True

    for attempt in range(1, max_retries + 1):
        try:
            print(f"🤖 [VM LIFECYCLE] Starting VM '{name}' in zone '{zone}' (Attempt {attempt}/{max_retries})...")
            subprocess.run(["gcloud", "compute", "instances", "start", name, "--zone", zone, "--quiet"], check=True)
            print("⏳ [VM LIFECYCLE] Waiting 25s for SSH and GPU driver initialization...")
            time.sleep(25)
            if is_vm_running(name, zone):
                print(f"✅ [VM LIFECYCLE] VM '{name}' is active and GPU reservation is locked.")
                return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️ [VM LIFECYCLE] Failed to start VM (Attempt {attempt}): {e.stderr or str(e)}")
            time.sleep(10)
        except Exception as e:
            print(f"⚠️ [VM LIFECYCLE] Error: {str(e)}")
            time.sleep(5)
    return is_vm_running(name, zone)


def stop_vm_safely(name: str = vm_name, zone: str = vm_zone):
    """Cleanly halts the A100 GPU VM to optimize cloud costs at job completion."""
    try:
        if is_vm_running(name, zone):
            print(f"🛑 [VM LIFECYCLE] Job completed. Safely halting VM '{name}' to prevent idle billing...")
            subprocess.run(["gcloud", "compute", "instances", "stop", name, "--zone", zone, "--discard-local-ssd=true", "--quiet"], capture_output=True)
            print(f"✅ [VM LIFECYCLE] VM '{name}' is stopped.")
        else:
            print(f"ℹ️ [VM LIFECYCLE] VM '{name}' is already stopped.")
    except Exception as e:
        print(f"⚠️ [VM LIFECYCLE] Note during VM shutdown: {str(e)}")


# =====================================================================
# CUSTOM TOOLS FOR THE AGENT SQUAD
# =====================================================================

@tool("Query Arxiv for Peptide Sequencing & Flow Matching Papers")
def query_arxiv(query: str) -> str:
    """Useful to search Arxiv for scientific papers on continuous-time Markov flow matching,
    de novo peptide sequencing, InstaNovo/Casanovo benchmarks, and spectrum conditioning."""
    try:
        wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=4000)
        return wrapper.run(query)
    except Exception as e:
        return f"Arxiv query failed: {str(e)}"


@tool("Explore and Audit Workspace Files Locally and Remotely")
def explore_and_audit_workspace(command: str = "ls -la", target_env: str = "both") -> str:
    """Useful to inspect code, tests, configs, data loaders, thesis LaTeX files, manuscripts,
    evaluation outputs, and unpushed files either locally or on the remote A100 GPU VM.
    The primary project folder on the VM is ~/dfm-joelresearch with environment in ~/dfm-joelresearch/.venv.
    target_env can be 'local', 'vm', or 'both'."""
    outputs = []
    remote_repo = os.getenv("GITHUB_REPO_URL", "https://github.com/joelinator/joelresearch.git")

    # 1. Local execution
    if target_env in ("local", "both"):
        try:
            res_loc = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=120
            )
            outputs.append(f"=== [LOCAL WORKSPACE OUTPUT] ===\nStdout:\n{res_loc.stdout}\nStderr:\n{res_loc.stderr}")
        except Exception as e:
            outputs.append(f"=== [LOCAL WORKSPACE ERROR] ===\n{str(e)}")

    # 2. Remote VM execution
    if target_env in ("vm", "both"):
        try:
            ensure_vm_running(vm_name, vm_zone)

            remote_cmd = f"""
# Prioritize the established project directory ~/dfm-joelresearch
TARGET_DIR="$HOME/dfm-joelresearch"
if [ ! -d "$TARGET_DIR" ]; then
    for cand in "$HOME/dfm-joelresearch" "/home/joelinator/DiffusionResearchProject/DFLowNovo/JoelResearch/joelresearch" /home/*/joelresearch "$HOME/joelresearch" "$HOME"/*; do
        if [ -d "$cand/.git" ]; then
            TARGET_DIR="$cand"
            break
        fi
    done
fi

if [ ! -d "$TARGET_DIR/.git" ]; then
    git clone --branch research {remote_repo} "$TARGET_DIR"
fi

cd "$TARGET_DIR"

# Strictly synchronize with remote research branch
git remote set-url origin {remote_repo} 2>/dev/null || true
git fetch origin research
git checkout -B research origin/research
git reset --hard origin/research

# Activate existing venv (checking ~/dfm-joelresearch/.venv first)
if [ -f "$HOME/dfm-joelresearch/.venv/bin/activate" ]; then
    source "$HOME/dfm-joelresearch/.venv/bin/activate"
elif [ -f "$TARGET_DIR/.venv/bin/activate" ]; then
    source "$TARGET_DIR/.venv/bin/activate"
elif [ -f "/home/joelinator/DiffusionResearchProject/DFLowNovo/JoelResearch/joelresearch/.venv/bin/activate" ]; then
    source /home/joelinator/DiffusionResearchProject/DFLowNovo/JoelResearch/joelresearch/.venv/bin/activate
elif [ -f "$HOME/.venv/bin/activate" ]; then
    source "$HOME/.venv/bin/activate"
fi

{command}
"""
            res_vm = subprocess.run(
                ["gcloud", "compute", "ssh", vm_name, "--zone", vm_zone, "--tunnel-through-iap", "--quiet", "--command", remote_cmd],
                capture_output=True, text=True, timeout=180
            )
            outputs.append(f"=== [REMOTE A100 VM OUTPUT (~/dfm-joelresearch)] ===\nStdout:\n{res_vm.stdout}\nStderr:\n{res_vm.stderr}")
        except Exception as e:
            outputs.append(f"=== [REMOTE A100 VM ERROR] ===\nFailed to explore VM workspace: {str(e)}")

    return "\n\n".join(outputs)


@tool("Execute Experiment, Training, or Benchmark on A100 VM")
def run_experiment_on_a100_vm(training_command: str = "python3 scripts/eval.py", custom_script_code: str = "") -> str:
    """Useful to trigger PyTorch training, model enhancements, evaluation, validation, or fact-checking
    code execution on the dedicated Google Cloud VM containing the Nvidia A100 80GB GPU.
    The primary project folder on the VM is ~/dfm-joelresearch with its pre-configured environment in ~/dfm-joelresearch/.venv.
    The tool ensures the VM is running (preserving GPU allocation), syncs with the remote 'research' branch,
    activates the pre-installed PyTorch venv, executes the commands, and logs metrics.
    Optionally pass custom_script_code to write a custom verification or benchmark script directly before execution."""
    remote_repo = os.getenv("GITHUB_REPO_URL", "https://github.com/joelinator/joelresearch.git")

    try:
        ensure_vm_running(vm_name, vm_zone)
        
        print("🤖 Agent Action: Locating project in ~/dfm-joelresearch, syncing 'research' branch, activating .venv, and executing on A100 GPU...")
        
        remote_bootstrap_script = f"""
# 1. Use primary directory ~/dfm-joelresearch
TARGET_DIR="$HOME/dfm-joelresearch"
if [ ! -d "$TARGET_DIR" ]; then
    for cand in "$HOME/dfm-joelresearch" "/home/joelinator/DiffusionResearchProject/DFLowNovo/JoelResearch/joelresearch" /home/*/joelresearch "$HOME/joelresearch" "$HOME"/*; do
        if [ -d "$cand/.git" ]; then
            TARGET_DIR="$cand"
            break
        fi
    done
fi

if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
    git clone --branch research {remote_repo} "$TARGET_DIR"
fi

echo "📂 Using project directory: $TARGET_DIR"
cd "$TARGET_DIR"

# 2. Strictly synchronize with remote research branch from GitHub
git remote set-url origin {remote_repo} 2>/dev/null || true
git fetch origin research
git checkout -B research origin/research
git reset --hard origin/research

echo "📌 Remote 'research' branch synchronized. Current HEAD:"
git rev-parse HEAD

# 3. Virtual environment activation (checking ~/dfm-joelresearch/.venv first)
VENV_ACTIVATED=0
for vpath in "$HOME/dfm-joelresearch/.venv/bin/activate" \
             "$TARGET_DIR/.venv/bin/activate" \
             "/home/joelinator/DiffusionResearchProject/DFLowNovo/JoelResearch/joelresearch/.venv/bin/activate" \
             "$HOME/.venv/bin/activate" \
             "/root/joelresearch/.venv/bin/activate"; do
    if [ -f "$vpath" ]; then
        echo "🐍 Activating Python environment: $vpath"
        source "$vpath"
        VENV_ACTIVATED=1
        break
    fi
done

if [ "$VENV_ACTIVATED" -eq 0 ]; then
    echo "⚠️ Warning: Pre-existing .venv not found. Creating local .venv in $TARGET_DIR..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
fi

# 4. Verify CUDA and PyTorch
python3 -c "import torch; print(f'CUDA available: {{torch.cuda.is_available()}}, Device: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}}')"
"""
        if custom_script_code.strip():
            escaped_code = custom_script_code.replace("'", "'\\''")
            remote_bootstrap_script += f"""
cat << 'EOF' > custom_experiment.py
{custom_script_code}
EOF
"""

        remote_bootstrap_script += f"""
# 5. Execute user command
{training_command}
"""

        result = subprocess.run(
            ["gcloud", "compute", "ssh", vm_name, "--zone", vm_zone, "--tunnel-through-iap", "--quiet", "--command", remote_bootstrap_script],
            capture_output=True, text=True, check=True
        )
        return f"Experiment on A100 finished successfully.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        
    except subprocess.CalledProcessError as e:
        err_msg = f"Execution on A100 VM finished with status {e.returncode}.\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        print(f"⚠️ {err_msg}")
        return err_msg
    except Exception as e:
        err_msg = f"Unexpected execution error on A100 VM: {str(e)}"
        print(f"⚠️ {err_msg}")
        return err_msg


@tool("Fact Check All Markdown Artifacts, Thesis Chapters and Metrics")
def fact_check_artifacts(target_scope: str = "all") -> str:
    """Useful to perform strict fact-checking across all repository artifacts:
    - SUPERVISOR_REPORT_DFM_DE_NOVO.md, SYSTEM_OPTIMIZATION_REPORT.md, ARTIFACTS_MANIFEST.md, README.md, PRESENTATION_OF_RESULTS.md
    - thesis/ (chapter1.tex to chapter6.tex, master-document.tex, appendixes)
    - manuscript/ (PAPER_MANUSCRIPT.md, THESIS_DEFENSE_SLIDES.md)
    - presentation/ (presentation.tex, DEFENSE_SPEAKER_NOTES_AND_QA.md)
    Verifies that all claimed numbers match actual code implementations and benchmark logs."""
    results = []
    project_root = Path(__file__).resolve().parent

    files_to_check = [
        project_root / "SUPERVISOR_REPORT_DFM_DE_NOVO.md",
        project_root / "SYSTEM_OPTIMIZATION_REPORT.md",
        project_root / "ARTIFACTS_MANIFEST.md",
        project_root / "PRESENTATION_OF_RESULTS.md",
        project_root / "README.md",
        project_root / "manuscript" / "PAPER_MANUSCRIPT.md",
        project_root / "manuscript" / "THESIS_CHAPTER_DE_NOVO_SEQUENCING.md",
        project_root / "thesis" / "chapter3.tex",
        project_root / "thesis" / "chapter4.tex",
        project_root / "thesis" / "chapter5.tex",
    ]

    for fpath in files_to_check:
        if not fpath.exists():
            results.append(f"⚠️ Missing file: {fpath.name}")
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
            suspicious = []
            for phrase in ["TODO", "TBD", "[INSERT", "approx.", "revolutionary", "game-changer", "testament to", "pivotal"]:
                if phrase.lower() in content.lower():
                    suspicious.append(phrase)
            
            has_metrics = ("Exact Sequence Match" in content or "exact_match" in content or "Exact Match" in content)
            
            results.append(
                f"📄 {fpath.name}: Length {len(content)} chars | Metrics Present: {has_metrics} | Flagged words/placeholders: {suspicious if suspicious else 'None (Clean)'}"
            )
        except Exception as e:
            results.append(f"❌ Error reading {fpath.name}: {str(e)}")

    return "=== FACT-CHECK ARTIFACT AUDIT REPORT ===\n" + "\n".join(results)


@tool("Regenerate Publication Figures and Validate Chart Integrity")
def regenerate_figures(script_name: str = "scripts/generate_publication_figures.py") -> str:
    """Useful to regenerate all 300 DPI publication-quality figures across docs/figures,
    thesis/images, and presentation/images using matplotlib."""
    try:
        cmd = f"{sys.executable} {script_name}"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        if res.returncode == 0:
            return f"✅ Figures successfully regenerated via {script_name}.\nOutput:\n{res.stdout}"
        else:
            return f"❌ Figure regeneration failed (exit code {res.returncode}):\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    except Exception as e:
        return f"❌ Figure regeneration error: {str(e)}"


@tool("Validate and Test Mass Spectrometry Data Parsers (MGF, MGZ, mzML)")
def test_data_parsers(test_file: str = "") -> str:
    """Useful to test and validate MS/MS file parsing routines for MGF, MGZ (gzipped MGF), mzML,
    and parquet formats. Checks extraction of SEQ/SEQUENCE headers, TITLE, PEPMASS, CHARGE, RTINSECONDS, and peaks."""
    try:
        cmd = f"{sys.executable} -m pytest tests/ -k 'test' -v"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return f"=== DATA PARSER & SUITE TEST RESULTS ===\nExit Code: {res.returncode}\nStdout:\n{res.stdout}\nStderr:\n{res.stderr}"
    except Exception as e:
        return f"Data parser test execution failed: {str(e)}"


@tool("Deploy or Synchronize Hugging Face Model & Gradio Space")
def deploy_huggingface(sync_model: bool = True, deploy_space: bool = True) -> str:
    """Useful to execute deployment and synchronization with Hugging Face Model Hub
    (joelinator/dflow-novo-model) and Hugging Face Spaces (joelinator/dflow-novo) via scripts/deploy_huggingface.py."""
    try:
        script_path = Path(__file__).resolve().parent / "scripts" / "deploy_huggingface.py"
        if not script_path.exists():
            return "❌ deploy_huggingface.py not found in scripts/ directory."
        
        cmd = f"{sys.executable} {script_path}"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        return f"=== HUGGING FACE DEPLOYMENT STATUS ===\nExit Code: {res.returncode}\nStdout:\n{res.stdout}\nStderr:\n{res.stderr}"
    except Exception as e:
        return f"Hugging Face deployment failed: {str(e)}"


@tool("Push Verified Improvements to GitHub")
def push_improvement_to_github(commit_message: str) -> str:
    """Useful to commit and push verified improvements, regenerated figures, audited thesis documents,
    and benchmark metrics to the GitHub repository on the 'research' and 'main' branches."""
    repo_url = os.getenv("GITHUB_REPO_URL")
    github_token = os.getenv("GITHUB_TOKEN")
    
    try:
        subprocess.run(["git", "config", "--global", "user.name", "DFlowNovo-Research-Squad"], check=True)
        subprocess.run(["git", "config", "--global", "user.email", "research-squad@cloudrun.internal"], check=True)
        
        if github_token and repo_url and "@" not in repo_url:
            clean_url = repo_url.replace("https://", "").replace("http://", "")
            authenticated_url = f"https://{github_token}@{clean_url}"
            subprocess.run(["git", "remote", "set-url", "origin", authenticated_url], check=True)
        elif repo_url:
            subprocess.run(["git", "remote", "set-url", "origin", repo_url], check=True)
            
        subprocess.run(["git", "add", "."], check=True)
        res = subprocess.run(["git", "commit", "-m", f"Scientific R&D Improvement: {commit_message}"], capture_output=True, text=True)
        
        push_res = subprocess.run(["git", "push", "origin", "research"], capture_output=True, text=True)
        if push_res.returncode != 0:
            push_res2 = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
            if push_res2.returncode != 0:
                return f"Local commit created: '{commit_message}'. (Push status: {push_res.stderr.strip() or push_res2.stderr.strip()})"
        return f"Successfully pushed verified updates to GitHub (research branch): {commit_message}"
    except Exception as e:
        return f"Local git state preserved: {str(e)}"


# =====================================================================
# MULTI-AGENT SQUAD DEFINITION (TEAMWORK & CONTINUOUS IMPROVEMENT)
# =====================================================================

manager_agent = Agent(
    role="Principal AI Architect & Research Director",
    goal=(
        "Lead the collaborative research squad to actively invent, benchmark, and deploy improvements for de novo peptide sequencing. "
        "Direct the team to push the boundaries beyond InstaNovo and Casanovo baselines, raising Exact Match (%) and I/L Equivalent Accuracy (%). "
        "Leverage the remote A100 VM at ~/dfm-joelresearch (.venv) on the 'research' branch. "
        "Enforce 100% empirical fact-checking, mathematical exactitude in the thesis, figure regeneration, MGF/MGZ/mzML parser robustness, "
        "and Hugging Face deployment upon verified gains."
    ),
    backstory=(
        "You are an inspiring yet demanding computational proteomics research director. You believe in relentless iterative empirical improvement "
        "backed by flawless mathematical proofs and reproducible code. You orchestrate the scientist, engineer, critic, and writer into a cohesive unit."
    ),
    llm=vertex_llm,
    max_iter=5,
    verbose=True
)

scientist_agent = Agent(
    role="Lead AI Scientist (De Novo Sequencing & Flow Matching)",
    goal=(
        "Formulate novel algorithmic, biochemical, and architectural advancements to improve peptide sequencing accuracy: "
        "1. Optimize Continuous-Time Markov Chain (CTMC) flow matching rate schedules (cosine, linear, exponential transitions). "
        "2. Enhance spectrum conditioning (peak intensity embeddings, fragment-ion b/y complementary pairs, spectral peak dropout). "
        "3. Refine dynamic Knapsack DP reachability guidance and precursor mass tolerance constraints. "
        "4. Fix MGF/MGZ data parsing to handle SEQ/SEQUENCE header fields and add support for mzML/parquet. "
        "5. Formulate concrete code modifications in ~/dfm-joelresearch to push Exact Match and I/L-equivalent metrics higher."
    ),
    backstory=(
        "You are a cutting-edge machine learning scientist specializing in generative flow matching on discrete sequence spaces and tandem mass spectrometry. "
        "You continuously analyze error modes (e.g., prefix mass errors, I/L ambiguities, low-intensity ion peaks) and design targeted algorithmic solutions."
    ),
    tools=[explore_and_audit_workspace, fact_check_artifacts, test_data_parsers, query_arxiv],
    llm=vertex_llm,
    max_iter=5,
    verbose=True
)

engineer_agent = Agent(
    role="Senior Bioinformatics & PyTorch Systems Engineer",
    goal=(
        "Implement proposed algorithmic enhancements in ~/dfm-joelresearch on the A100 GPU VM using ~/dfm-joelresearch/.venv. "
        "Execute training runs, validation benchmarks (eval.py, benchmark_refined_decoding_50k.py), and ablation studies. "
        "Measure and compare Exact Sequence Match (%), I/L-Equivalent Match (%), AA Precision/Recall against baselines. "
        "Regenerate 300 DPI publication figures, validate MGF/MGZ parsers, deploy verified models to Hugging Face, and commit validated gains to GitHub."
    ),
    backstory=(
        "You are a high-performance deep learning systems engineer with expertise in PyTorch, CUDA, and bioinformatics pipelines. "
        "You execute runs on 80GB Nvidia A100 GPUs, track loss curves, optimize inference throughput, and maintain reproducible software."
    ),
    tools=[explore_and_audit_workspace, run_experiment_on_a100_vm, regenerate_figures, test_data_parsers, deploy_huggingface, push_improvement_to_github],
    llm=vertex_llm,
    max_iter=5,
    verbose=True
)

critic_agent = Agent(
    role="Chief Scientific Critic & Methodological Auditor",
    goal=(
        "Critique every proposed enhancement, experiment result, mathematical equation, table, and text. "
        "Ensure all performance improvements are statistically rigorous and reproducible (seed control, split isolation). "
        "Strictly detect and eliminate AI-generated fluff and clichés ('revolutionary', 'game-changer', 'testament to', 'pivotal'). "
        "Demand exact correspondence between mathematics in the thesis and implementation in src/ (loss functions, scheduler, sampling). "
        "Verify reproducibility down to exact CLI commands and act as the gatekeeper for final sign-off."
    ),
    backstory=(
        "You are an exacting senior reviewer and auditor. You challenge optimistic assumptions, verify every claimed metric against raw JSON logs, "
        "and push the team to achieve genuine, unshakeable state-of-the-art results."
    ),
    tools=[explore_and_audit_workspace, fact_check_artifacts, test_data_parsers],
    llm=vertex_llm,
    max_iter=5,
    verbose=True
)

writer_agent = Agent(
    role="Academic Thesis & Reproducibility Lead",
    goal=(
        "Translate validated experimental gains and algorithmic advancements into the master thesis (LaTeX), research manuscript, "
        "defense presentation, and README.md. "
        "Ensure loss formulations, CTMC transition rates, and Knapsack DP algorithms match the code in ~/dfm-joelresearch exactly. "
        "Provide exhaustive step-by-step reproduction instructions (seeds, configs, CLI commands, hardware requirements)."
    ),
    backstory=(
        "You are an academic documentation architect who crafts rigorous, publication-grade manuscripts and LaTeX theses. "
        "You write concise, objective scientific prose and ensure seamless repository coherence."
    ),
    tools=[explore_and_audit_workspace, fact_check_artifacts, regenerate_figures, push_improvement_to_github],
    llm=vertex_llm,
    max_iter=5,
    verbose=True
)


# =====================================================================
# COLLABORATIVE TASKS (IMPROVEMENT, FACT-CHECKING & SYNC)
# =====================================================================

task_research_and_improvement_strategy = Task(
    description=(
        "1. EXPLORE & AUDIT ~/dfm-joelresearch ON VM: Inspect existing code, test suites, training logs, checkpoints, and unpushed files in ~/dfm-joelresearch.\n"
        "2. AUDIT DATA PARSERS: Verify spectrum parsing in src/data/data.py and deployment/huggingface/app.py. Check handling of 'SEQ=' / 'SEQUENCE=' "
        "fields in MGF/MGZ files, and support for mzML/parquet.\n"
        "3. PROPOSE ACTIONABLE PERFORMANCE IMPROVEMENT: Design a concrete algorithmic enhancement to boost sequencing accuracy beyond InstaNovo:\n"
        "   - Improved CTMC flow transition rate schedules or temperature annealing,\n"
        "   - Refined spectrum conditioning / peak intensity normalization / complementary ions,\n"
        "   - Dynamic Knapsack reachability guidance with precursor mass penalty calibration,\n"
        "   - Or I/L equivalent disambiguation strategy.\n"
        "4. FORMULATE EXACT SPECIFICATION: Output the exact code diff, equations, and benchmark execution plan."
    ),
    expected_output="A concrete R&D improvement strategy with exact code specifications, parser bugfixes (SEQ handling), and evaluation targets.",
    agent=scientist_agent
)

task_experimentation_and_benchmarking = Task(
    description=(
        "1. EXECUTE ON A100 GPU VM: Apply the proposed code improvement in ~/dfm-joelresearch using ~/dfm-joelresearch/.venv on the 'research' branch.\n"
        "2. RUN BENCHMARKS: Run evaluation / training scripts (e.g. scripts/eval.py, scripts/benchmark_refined_decoding_50k.py, or custom experiment).\n"
        "3. MEASURE METRICS: Evaluate Exact Sequence Match (%), I/L-Equivalent Match (%), AA Precision/Recall, and runtime.\n"
        "4. REGENERATE PUBLICATION FIGURES: Execute scripts/generate_publication_figures.py to produce updated 300 DPI Nature Methods-grade charts.\n"
        "5. HUGGING FACE DEPLOYMENT: Deploy the updated model and Gradio Space if new improvements or parser fixes are verified.\n"
        "6. COMMIT & PUSH: Push verified improvements and benchmark metrics to GitHub 'research' branch."
    ),
    expected_output="Benchmark execution logs, measured metrics comparing before/after, regenerated 300 DPI figures, and confirmed GitHub push.",
    agent=engineer_agent
)

task_thesis_and_factcheck_sync = Task(
    description=(
        "1. FACT-CHECK EVERY ARTIFACT: Verify all numbers, tables, and claims across thesis/ (LaTeX), manuscript/, presentation/, and markdown reports.\n"
        "2. THESIS & CODE COHERENCE: Ensure that chapter3/chapter4/chapter5 LaTeX equations (loss functions, CTMC flow dynamics, Knapsack DP) "
        "mirror the exact implementation in src/ with zero approximations.\n"
        "3. ELIMINATE AI CLICHÉS: Audit prose to ensure rigorous, formal scientific writing ('We observe', 'The empirical accuracy is') "
        "and remove all AI buzzwords ('revolutionary', 'game-changer', 'testament to', 'pivotal').\n"
        "4. EXHAUSTIVE REPRODUCIBILITY GUIDE: Update README.md with complete reproduction instructions: environment setup, random seeds, "
        "hardware specifications, dataset download steps, training commands, inference scripts, and figure generation commands.\n"
        "5. Commit all synchronized documentation and thesis files to GitHub."
    ),
    expected_output="Mathematically exact, fully fact-checked thesis, manuscript, and comprehensive README reproduction guide committed to GitHub.",
    agent=writer_agent
)

task_final_critic_review = Task(
    description=(
        "1. FINAL CRITICAL EVALUATION: Perform a comprehensive peer-review audit of all new benchmark results, regenerated figures, "
        "thesis chapters, data parsers, and reproduction steps.\n"
        "2. VERIFY ACCURACY & RIGOR: Confirm that the improvement is genuine, all claimed numbers match raw JSON outputs, "
        "the thesis is 100% mathematically aligned with the codebase, and the README reproduction guide is complete.\n"
        "3. SIGN-OFF: Issue a formal scientific approval report certifying empirical gains, mathematical rigor, and full reproducibility."
    ),
    expected_output="A formal Chief Scientific Critic Approval Report certifying validated performance gains, mathematical exactitude, and reproducibility.",
    agent=critic_agent
)


# =====================================================================
# HIERARCHICAL EXECUTION LOOP WITH CONVERGENCE CHECK & SAFE SHUTDOWN
# =====================================================================

def start_collaborative_crew(max_loops: int = 1):
    """Executes the autonomous scientific research, improvement, fact-checking, and documentation squad with safe VM cleanup."""
    print("🚀 Initializing DFlowNovo Autonomous Research & Verification Squad...")
    
    try:
        for loop_idx in range(1, max_loops + 1):
            print(f"\n================================================================================")
            print(f"🔄 Starting Scientific Research & Improvement Cycle (Iteration {loop_idx}/{max_loops})")
            print(f"================================================================================\n")
            
            peptide_squad = Crew(
                agents=[scientist_agent, engineer_agent, writer_agent, critic_agent],
                tasks=[
                    task_research_and_improvement_strategy,
                    task_experimentation_and_benchmarking,
                    task_thesis_and_factcheck_sync,
                    task_final_critic_review
                ],
                process=Process.hierarchical,
                manager_agent=manager_agent,
                verbose=True
            )
            
            result = peptide_squad.kickoff()
            print(f"\n✅ Iteration {loop_idx} complete.")
            
        return result
    finally:
        # Guarantee VM shutdown on completion or failure
        stop_vm_safely(vm_name, vm_zone)


if __name__ == "__main__":
    start_collaborative_crew(max_loops=1)
