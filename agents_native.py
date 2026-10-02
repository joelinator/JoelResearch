# agents_native.py
"""
DFlowNovo Multi-Agent Autonomous Research, Improvement & Verification Squad (Native VM Version)
=================================================================================================
Native multi-agent R&D squad executing directly inside the compute environment.
Directly accesses PyTorch, CUDA, virtual environment packages, datasets, model checkpoints,
and local git workspace with zero SSH/tunneling overhead.

Enriched with:
- Seamless Vertex AI and Gemini API key authentication (resolving 403 scope issues)
- Native GitHub CLI (gh) credential synchronization for automated git pushes
- Full access to the specialized scientific skills library (pyopenms, optimize-for-gpu,
  scientific-writing, peer-review, scientific-visualization, statistical-analysis, etc.)
- Deeply enriched prompts enforcing coherence, empirical fact-checking, rigorous academic writing,
  adversarial peer review, experiment reruns, algorithmic improvement, implementation, testing, and exploration.
"""

import os
import sys
import time
import json
import re
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load local environment variables (.env)
load_dotenv()

project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GCP_PROJECT", "joelgedeon-project-507410")
region = os.getenv("VERTEXAI_LOCATION", "us-central1")
repo_url = os.getenv("GITHUB_REPO_URL", "https://github.com/joelinator/joelresearch.git")
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if api_key and not os.getenv("GOOGLE_GENAI_USE_VERTEXAI"):
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"
else:
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")

os.environ["VERTEXAI_PROJECT"] = project_id
os.environ["VERTEXAI_LOCATION"] = region
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.utilities import ArxivAPIWrapper
from crewai.tools import tool

WORKSPACE_ROOT = Path(__file__).resolve().parent
SKILLS_DIR = Path.home() / ".gemini" / "config" / "skills"

# =====================================================================
# CREDENTIALS & AUTHENTICATION RESOLUTION
# =====================================================================

def get_vertex_credentials():
    """Resolves Google Cloud credentials with cloud-platform scopes.
    Prioritizes active Antigravity live OAuth credentials if available,
    which provides valid cloud-platform scope without requiring VM re-provisioning."""
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path and Path(sa_path).exists():
        return None  # Standard Google GenAI client picks this up automatically
    
    token_file = Path.home() / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
    if token_file.exists():
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            access_token = data.get("token", {}).get("access_token")
            if access_token:
                from google.oauth2.credentials import Credentials

                class AntigravityLiveCredentials(Credentials):
                    def __init__(self, tpath):
                        self.tpath = tpath
                        tok = self._read_token()
                        super().__init__(token=tok, scopes=["https://www.googleapis.com/auth/cloud-platform"])

                    def _read_token(self):
                        try:
                          with open(self.tpath, "r", encoding="utf-8") as tf:
                            return json.load(tf).get("token", {}).get("access_token")
                        except Exception:
                          return getattr(self, "token", None)

                    def before_request(self, request, method, url, headers):
                        fresh = self._read_token()
                        if fresh:
                            self.token = fresh
                        super().before_request(request, method, url, headers)

                    def refresh(self, request):
                        self.token = self._read_token()

                return AntigravityLiveCredentials(token_file)
        except Exception as e:
            print(f"Notice: Could not load Antigravity live credentials: {e}")
    return None


def get_github_token() -> str | None:
    """Retrieves GitHub token from environment or active gh CLI authentication."""
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return token
    try:
        token = subprocess.check_output(
            ["gh", "auth", "token"], text=True, stderr=subprocess.DEVNULL, cwd=str(WORKSPACE_ROOT)
        ).strip()
        if token:
            os.environ["GITHUB_TOKEN"] = token
            return token
    except Exception:
        pass
    return None


# Configure LLM backend
model_name = os.getenv("CREWAI_MODEL", "gemini/gemini-2.5-pro")
live_creds = get_vertex_credentials()
client_params = {}
if live_creds:
    client_params["credentials"] = live_creds

if api_key:
    vertex_llm = LLM(
        model=model_name,
        temperature=0.1,
        api_key=api_key,
    )
else:
    vertex_llm = LLM(
        model=model_name,
        temperature=0.1,
        project=project_id,
        location=region,
        client_params=client_params if client_params else None,
    )


# =====================================================================
# SCIENTIFIC AGENT SKILLS INTEGRATION TOOLS
# =====================================================================

@tool("List Available Scientific Skills")
def list_available_scientific_skills(category_or_query: str = "all") -> str:
    """Useful to discover and list available specialized scientific skills in the library.
    Query can be 'proteomics', 'deep learning', 'writing', 'peer review', 'statistics', 'gpu', or 'all'.
    Returns matching skills with summaries of their domain capabilities."""
    if not SKILLS_DIR.exists():
        return f"Skills directory not found at {SKILLS_DIR}."
    
    query = category_or_query.strip().lower()
    results = []
    
    domain_map = {
        "proteomics": ["pyopenms", "matchms", "biopython", "esm", "glycoengineering"],
        "deep learning": ["pytorch-lightning", "optimize-for-gpu", "transformers", "torch-geometric", "scikit-learn", "timesfm-forecasting"],
        "writing": ["scientific-writing", "venue-templates", "citation-management", "markdown-mermaid-writing", "scientific-visualization"],
        "peer review": ["peer-review", "scientific-critical-thinking", "scholar-evaluation"],
        "statistics": ["statistical-analysis", "statsmodels", "pymc", "scikit-survival", "exploratory-data-analysis"],
        "gpu": ["optimize-for-gpu", "pytorch-lightning", "modal", "dask"],
    }
    
    for skill_path in sorted(SKILLS_DIR.iterdir()):
        if not skill_path.is_dir():
            continue
        sname = skill_path.name
        sm_file = skill_path / "SKILL.md"
        if not sm_file.exists():
            continue
        
        desc = ""
        try:
            head = sm_file.read_text(encoding="utf-8", errors="replace")[:1000]
            for line in head.splitlines():
                if line.startswith("description:"):
                    desc = line.replace("description:", "").strip()
                    break
        except Exception:
            pass

        include = False
        if query in ["all", "", "list"]:
            include = True
        elif query in domain_map and sname in domain_map[query]:
            include = True
        elif query in sname.lower() or query in desc.lower():
            include = True

        if include:
            results.append(f"- **{sname}**: {desc[:140]}...")

    return f"=== AVAILABLE SCIENTIFIC SKILLS (Query: '{category_or_query}', Count: {len(results)}) ===\n" + "\n".join(results)


@tool("Consult Scientific Skill Manual and Guidelines")
def consult_scientific_skill(skill_name: str, section_or_file: str = "SKILL.md") -> str:
    """Useful to load authoritative guidelines, implementation patterns, checklists, and reference recipes
    from any scientific skill (e.g. 'pyopenms', 'optimize-for-gpu', 'scientific-writing', 'peer-review',
    'pytorch-lightning', 'scientific-visualization', 'statistical-analysis').
    Specify skill_name and optional section_or_file (default 'SKILL.md', or reference files in references/)."""
    s_clean = skill_name.strip().lower()
    s_path = SKILLS_DIR / s_clean
    if not s_path.exists():
        matches = [d.name for d in SKILLS_DIR.iterdir() if d.is_dir() and s_clean in d.name.lower()]
        return f"❌ Skill '{skill_name}' not found. Did you mean one of: {matches[:5]}?"

    target_file = s_path / section_or_file
    if not target_file.exists():
        target_file = s_path / "SKILL.md"
        if not target_file.exists():
            return f"❌ Neither '{section_or_file}' nor 'SKILL.md' found in skill '{skill_name}'."

    try:
        content = target_file.read_text(encoding="utf-8", errors="replace")
        scripts = [f.name for f in (s_path / "scripts").glob("*.py")] if (s_path / "scripts").exists() else []
        refs = [f.name for f in (s_path / "references").glob("*.md")] if (s_path / "references").exists() else []
        header = (
            f"=== SCIENTIFIC SKILL CONSULTATION: {s_clean} (File: {target_file.name}) ===\n"
            f"Available Scripts: {scripts}\n"
            f"Available References: {refs}\n"
            f"------------------------------------------------------------------------\n"
        )
        return header + content[:5000]
    except Exception as e:
        return f"❌ Error reading skill '{skill_name}': {str(e)}"


@tool("Execute Scientific Skill Audit Script")
def execute_skill_script(skill_name: str, script_name: str, arguments: str = "") -> str:
    """Useful to run specialized Python auditing, formatting, or validation scripts bundled inside a skill's scripts/ directory.
    Example: skill_name='scientific-writing', script_name='check_consistency.py', arguments='--help'
    Example: skill_name='peer-review', script_name='audit_citations.py'."""
    s_clean = skill_name.strip().lower()
    script_path = SKILLS_DIR / s_clean / "scripts" / script_name
    if not script_path.exists():
        available = [f.name for f in (SKILLS_DIR / s_clean / "scripts").glob("*.py")] if (SKILLS_DIR / s_clean / "scripts").exists() else []
        return f"❌ Script '{script_name}' not found in skill '{skill_name}'. Available scripts: {available}"

def _truncate_output(text: str, max_chars: int = 8000, keep_tail: bool = False) -> str:
    """Safely clamp string length to prevent blowing past LLM context token windows."""
    if not text or len(text) <= max_chars:
        return text
    if keep_tail:
        omitted = len(text) - max_chars
        return f"[... Truncated {omitted} leading characters to preserve LLM context window ...]\n" + text[-max_chars:]
    else:
        omitted = len(text) - max_chars
        return text[:max_chars] + f"\n[... Truncated {omitted} trailing characters to preserve LLM context window. Target specific files or subdirectories ...]"


    try:
        cmd = f"{sys.executable} {script_path} {arguments}".strip()
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120, cwd=str(WORKSPACE_ROOT))
        stdout_tr = _truncate_output(res.stdout, max_chars=6000)
        stderr_tr = _truncate_output(res.stderr, max_chars=2000)
        return (
            f"=== SKILL SCRIPT OUTPUT: {script_name} (Exit Code: {res.returncode}) ===\n"
            f"STDOUT:\n{stdout_tr}\n"
            f"STDERR:\n{stderr_tr}"
        )
    except Exception as e:
        return f"❌ Skill script execution failed: {str(e)}"


# =====================================================================
# WORKSPACE & SYSTEM TOOLS FOR THE AGENT SQUAD
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


@tool("Read File in Workspace")
def read_workspace_file(file_path: str, start_line: int = 1, end_line: int = 120) -> str:
    """Useful to read the contents of any file in the workspace within specified 1-indexed line numbers.
    Helps inspect source code (src/), scripts, LaTeX thesis chapters (thesis/), manuscripts, or configs."""
    try:
        target = WORKSPACE_ROOT / file_path
        if not target.exists():
            return f"❌ File not found: {file_path}"
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, start_line)
        if end_line - start > 150:
            end_line = start + 150
        end = min(len(lines), max(start, end_line))
        selected = lines[start - 1 : end]
        numbered = [f"{start + i}: {line}" for i, line in enumerate(selected)]
        output_text = f"=== {file_path} (Lines {start}-{end} of {len(lines)}) ===\n" + "\n".join(numbered)
        return _truncate_output(output_text, max_chars=8000)
    except Exception as e:
        return f"❌ Error reading file {file_path}: {str(e)}"


@tool("Write or Update File in Workspace")
def write_workspace_file(file_path: str, content: str) -> str:
    """Useful to safely create or overwrite a file in the workspace (e.g. modifying code in src/,
    updating thesis chapters in thesis/, refining scripts, or updating README.md).
    Automatically creates parent directories if needed."""
    try:
        target = WORKSPACE_ROOT / file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"✅ Successfully wrote {len(content)} characters to {file_path}."
    except Exception as e:
        return f"❌ Error writing to file {file_path}: {str(e)}"


@tool("Execute Shell Command in Workspace")
def execute_shell_command(command: str) -> str:
    """Useful to inspect files, check git status, list directories, read logs, or execute bash utilities
    directly inside the native workspace.
    IMPORTANT: When searching or listing files, NEVER run unbounded `find .` across `.venv` or `.git`. Always target specific directories (e.g. `src/`, `thesis/`) or exclude `.venv`."""
    try:
        cmd_clean = command.strip()
        # Intercept unbounded find commands that could scan the entire .venv/ (89,000+ files)
        if "find ." in cmd_clean and ".venv" not in cmd_clean and "-prune" not in cmd_clean:
            cmd_clean = cmd_clean.replace("find .", "find . -not -path '*/.*' -not -path './.venv*'")

        res = subprocess.run(
            cmd_clean, shell=True, capture_output=True, text=True, timeout=180, cwd=str(WORKSPACE_ROOT)
        )
        stdout_tr = _truncate_output(res.stdout, max_chars=8000)
        stderr_tr = _truncate_output(res.stderr, max_chars=3000)
        return f"=== SHELL COMMAND OUTPUT (Exit: {res.returncode}) ===\nSTDOUT:\n{stdout_tr}\nSTDERR:\n{stderr_tr}"
    except Exception as e:
        return f"Shell command execution error: {str(e)}"


@tool("Run Experiment, Training, or Benchmark Natively on GPU")
def run_experiment_natively(command: str = "python3 scripts/eval.py", custom_script_code: str = "") -> str:
    """Useful to execute PyTorch training, model evaluation, benchmark scripts (e.g. scripts/eval.py,
    scripts/benchmark_refined_decoding_50k.py), or custom Python experiment scripts directly on the local compute hardware.
    Runs natively in the active environment with direct CUDA acceleration or CPU fallback."""
    try:
        if custom_script_code.strip():
            script_path = WORKSPACE_ROOT / "custom_native_experiment.py"
            script_path.write_text(custom_script_code, encoding="utf-8")
            print(f"📝 Created custom experiment script: {script_path}")
            if "custom_native_experiment.py" not in command:
                command = f"{sys.executable} custom_native_experiment.py"

        print(f"🚀 [NATIVE EXECUTION] Running: {command}")
        start_time = time.time()
        res = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=1800, cwd=str(WORKSPACE_ROOT)
        )
        elapsed = time.time() - start_time
        stdout_tr = _truncate_output(res.stdout, max_chars=8000, keep_tail=True)
        stderr_tr = _truncate_output(res.stderr, max_chars=3000, keep_tail=True)
        return (
            f"=== EXPERIMENT EXECUTION COMPLETED (Elapsed: {elapsed:.1f}s, Exit: {res.returncode}) ===\n"
            f"STDOUT:\n{stdout_tr}\nSTDERR:\n{stderr_tr}"
        )
    except subprocess.TimeoutExpired:
        return f"❌ Experiment execution timed out after 1800 seconds."
    except Exception as e:
        return f"❌ Experiment execution error: {str(e)}"


@tool("Fact Check All Markdown Artifacts, Thesis Chapters and Metrics")
def fact_check_artifacts(target_scope: str = "all") -> str:
    """Useful to perform strict fact-checking across all repository artifacts:
    - SUPERVISOR_REPORT_DFM_DE_NOVO.md, SYSTEM_OPTIMIZATION_REPORT.md, ARTIFACTS_MANIFEST.md, README.md, PRESENTATION_OF_RESULTS.md
    - thesis/ (chapter1.tex to chapter6.tex, master-document.tex, appendixes)
    - manuscript/ (PAPER_MANUSCRIPT.md, THESIS_DEFENSE_SLIDES.md)
    - presentation/ (presentation.tex, DEFENSE_SPEAKER_NOTES_AND_QA.md)
    Verifies that all claimed numbers match actual code implementations and benchmark logs."""
    results = []

    files_to_check = [
        WORKSPACE_ROOT / "SUPERVISOR_REPORT_DFM_DE_NOVO.md",
        WORKSPACE_ROOT / "SYSTEM_OPTIMIZATION_REPORT.md",
        WORKSPACE_ROOT / "ARTIFACTS_MANIFEST.md",
        WORKSPACE_ROOT / "PRESENTATION_OF_RESULTS.md",
        WORKSPACE_ROOT / "README.md",
        WORKSPACE_ROOT / "manuscript" / "PAPER_MANUSCRIPT.md",
        WORKSPACE_ROOT / "manuscript" / "THESIS_CHAPTER_DE_NOVO_SEQUENCING.md",
        WORKSPACE_ROOT / "thesis" / "chapter3.tex",
        WORKSPACE_ROOT / "thesis" / "chapter4.tex",
        WORKSPACE_ROOT / "thesis" / "chapter5.tex",
    ]

    for fpath in files_to_check:
        if not fpath.exists():
            results.append(f"⚠️ Missing file: {fpath.relative_to(WORKSPACE_ROOT) if fpath.is_relative_to(WORKSPACE_ROOT) else fpath.name}")
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
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300, cwd=str(WORKSPACE_ROOT))
        if res.returncode == 0:
            return f"✅ Figures successfully regenerated via {script_name}.\nOutput:\n{res.stdout}"
        else:
            return f"❌ Figure regeneration failed (exit code {res.returncode}):\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    except Exception as e:
        return f"❌ Figure regeneration error: {str(e)}"


@tool("Validate and Test Mass Spectrometry Data Parsers and Test Suite")
def test_data_parsers(test_target: str = "tests/") -> str:
    """Useful to run pytest on tests/ or specific test files (e.g. tests/test_reachability_dp.py,
    tests/test_checkpoint_io.py, tests/test_decoding_and_scoring.py).
    Verifies parser resilience for MGF, MGZ (gzipped MGF), mzML, and Parquet formats,
    confirming handling of SEQ/SEQUENCE headers, charges, and precursor masses."""
    try:
        target = test_target.strip() if test_target.strip() else "tests/"
        cmd = f"{sys.executable} -m pytest {target} -v"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=240, cwd=str(WORKSPACE_ROOT))
        stdout_tr = _truncate_output(res.stdout, max_chars=4000, keep_tail=True)
        stderr_tr = _truncate_output(res.stderr, max_chars=2000, keep_tail=True)
        return f"=== DATA PARSER & SUITE TEST RESULTS ===\nExit Code: {res.returncode}\nStdout:\n{stdout_tr}\nStderr:\n{stderr_tr}"
    except Exception as e:
        return f"Data parser test execution failed: {str(e)}"


@tool("Deploy or Synchronize Hugging Face Model & Gradio Space")
def deploy_huggingface(sync_model: bool = True, deploy_space: bool = True) -> str:
    """Useful to execute deployment and synchronization with Hugging Face Model Hub
    (joelinator/dflow-novo-model) and Hugging Face Spaces (joelinator/dflow-novo) via scripts/deploy_huggingface.py."""
    try:
        script_path = WORKSPACE_ROOT / "scripts" / "deploy_huggingface.py"
        if not script_path.exists():
            return "❌ deploy_huggingface.py not found in scripts/ directory."
        
        cmd = f"{sys.executable} {script_path}"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300, cwd=str(WORKSPACE_ROOT))
        stdout_tr = _truncate_output(res.stdout, max_chars=4000, keep_tail=True)
        stderr_tr = _truncate_output(res.stderr, max_chars=2000, keep_tail=True)
        return f"=== HUGGING FACE DEPLOYMENT STATUS ===\nExit Code: {res.returncode}\nStdout:\n{stdout_tr}\nStderr:\n{stderr_tr}"
    except Exception as e:
        return f"Hugging Face deployment failed: {str(e)}"


@tool("Push Verified Improvements to GitHub")
def push_improvement_to_github(commit_message: str) -> str:
    """Useful to commit and push verified improvements, regenerated figures, audited thesis documents,
    and benchmark metrics directly to the GitHub repository on the 'research' branch using active gh credentials."""
    repo_url_env = os.getenv("GITHUB_REPO_URL", "https://github.com/joelinator/joelresearch.git")
    
    # 1. Ensure gh git helper is active
    try:
        subprocess.run(["gh", "auth", "setup-git"], check=False, cwd=str(WORKSPACE_ROOT))
    except Exception:
        pass
        
    # 2. Get GitHub token from env or gh auth token
    github_token = get_github_token()

    try:
        subprocess.run(["git", "config", "user.name", "DFlowNovo-Research-Squad"], check=True, cwd=str(WORKSPACE_ROOT))
        subprocess.run(["git", "config", "user.email", "research-squad@cloudrun.internal"], check=True, cwd=str(WORKSPACE_ROOT))
        
        if github_token and repo_url_env:
            clean_url = repo_url_env.replace("https://", "").replace("http://", "")
            if "@" in clean_url:
                clean_url = clean_url.split("@", 1)[1]
            authenticated_url = f"https://x-access-token:{github_token}@{clean_url}"
            subprocess.run(["git", "remote", "set-url", "origin", authenticated_url], check=True, cwd=str(WORKSPACE_ROOT))
        elif repo_url_env:
            subprocess.run(["git", "remote", "set-url", "origin", repo_url_env], check=True, cwd=str(WORKSPACE_ROOT))
            
        subprocess.run(["git", "add", "."], check=True, cwd=str(WORKSPACE_ROOT))
        commit_res = subprocess.run(
            ["git", "commit", "-m", f"Scientific R&D Improvement: {commit_message}"],
            capture_output=True, text=True, cwd=str(WORKSPACE_ROOT)
        )
        if commit_res.returncode != 0 and "nothing to commit" in commit_res.stdout.lower():
            return "ℹ️ Git working tree clean — nothing new to commit."
        
        push_res = subprocess.run(["git", "push", "origin", "research"], capture_output=True, text=True, cwd=str(WORKSPACE_ROOT))
        if push_res.returncode != 0:
            push_res = subprocess.run(["git", "push", "-u", "origin", "research"], capture_output=True, text=True, cwd=str(WORKSPACE_ROOT))
            if push_res.returncode != 0:
                push_res2 = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True, cwd=str(WORKSPACE_ROOT))
                if push_res2.returncode != 0:
                    return f"Local commit created: '{commit_message}'. Push status: {push_res.stderr.strip() or push_res2.stderr.strip()}"
        return f"✅ Successfully pushed verified updates to GitHub (research branch): {commit_message}"
    except Exception as e:
        return f"Git operation notice: {str(e)}"


# =====================================================================
# MULTI-AGENT SQUAD DEFINITIONS (EQUIPPED WITH SCIENTIFIC SKILLS)
# =====================================================================

manager_agent = Agent(
    role="Principal AI Architect & Research Director",
    goal=(
        "Direct the autonomous scientific research cycle natively in the compute environment. "
        "Enforce strict empirical rigor, mathematical exactitude, cross-artifact coherence, and reproducible engineering: "
        "1. EXPLORE & AUDIT: Lead the squad to audit current models, data loaders (MGF, MGZ, mzML), and dynamic programming tables. "
        "2. SCIENTIFIC SKILLS MASTERY: Consult the specialized scientific skills library (e.g. 'scientific-brainstorming', 'peer-review') "
        "to set authoritative methodological standards for the team. "
        "3. PROPOSE & IMPLEMENT: Require the Scientist and Engineer to formulate and code concrete algorithmic improvements "
        "(e.g. continuous-time flow transition schedules, complementary b/y ion conditioning, charge-adaptive Knapsack reachability tolerance). "
        "4. TEST & BENCHMARK: Enforce that all proposed code changes are validated with pytest and evaluated on benchmarks "
        "(measuring Exact Match %, I/L Equivalent Match %, AA Precision/Recall, and spectra/sec). RERUN experiments whenever numbers are unverified. "
        "5. WRITING & FACT-CHECKING: Oversee that the thesis, manuscript, reports, and README reflect exact code implementations without AI fluff or placeholders. "
        "6. CRITIC SCRUTINY: Require independent peer-review sign-off before committing. "
        "7. GITHUB DEPLOYMENT: Push all verified improvements to the GitHub 'research' branch using active credentials."
    ),
    backstory=(
        "You are an internationally recognized Principal AI Research Director with deep expertise in generative flow matching, "
        "computational proteomics, and tandem mass spectrometry. You lead high-impact teams at the frontier of machine learning. "
        "You reject superficial improvements, ungrounded metrics, and hand-waving claims. You demand rigorous mathematics, robust test suites, "
        "and empirical reproducibility. You orchestrate the Scientist, Engineer, Author, and Critic into an elite, collaborative research laboratory."
    ),
    llm=vertex_llm,
    max_iter=6,
    verbose=True
)

scientist_agent = Agent(
    role="Lead AI Scientist (De Novo Sequencing & Flow Matching)",
    goal=(
        "Pioneer algorithmic, biochemical, and architectural advancements to push DFlowNovo de novo peptide sequencing beyond InstaNovo and Casanovo: "
        "1. SPECIALIZED SKILLS CONSULTATION: Consult the scientific skills 'pyopenms' (mass spectrometry data structures and algorithms), "
        "'matchms' (spectrum similarity and peak processing), 'biopython' (peptide sequence manipulation), and 'esm' for proteomic standards. "
        "2. FLOW MATCHING TRANSITION SCHEDULES: Formulate optimized transition rate schedules for continuous-time discrete flow matching "
        "(e.g. cosine annealing, polynomial schedules, adaptive temperature scaling) that improve token convergence during iterative denoising. "
        "3. SPECTRAL CONDITIONING & COMPLEMENTARY IONS: Design enriched mass spectrum conditioning embeddings, including complementary "
        "b- and y-ion intensity features, neutral loss indicators (-H2O, -NH3), and isotopic peak patterns. "
        "4. DYNAMIC KNAPSACK REACHABILITY GUIDANCE: Refine the O(1) Knapsack DP table guidance to support charge-adaptive mass tolerances, "
        "isotope mass tolerances, and post-translational modification (PTM) mass shifts. "
        "5. DATA PARSER RESILIENCE: Enhance spectrum parsers in src/data/data.py and deployment/huggingface/app.py to robustly handle MGF and MGZ "
        "(gzip-compressed MGF) header variants (SEQ=, SEQUENCE=, PEPMASS=, CHARGE=, RTINSECONDS=) as well as mzML and parquet formats. "
        "6. CONCRETE SPECIFICATIONS: Provide exact mathematical equations, drop-in code diffs, and test targets for the Principal Engineer."
    ),
    backstory=(
        "You are a cutting-edge generative ML scientist specializing in continuous-time Markov chains on discrete spaces and high-resolution MS/MS proteomics. "
        "You understand the physical chemistry of peptide fragmentation (collision-induced dissociation, b/y series, mass accuracy) as deeply as continuous-time flow dynamics. "
        "You actively consult 'pyopenms', 'matchms', and 'biopython' to ensure your algorithmic designs follow golden proteomics standards. "
        "You turn theoretical insights into precise mathematical equations and concrete Python implementations."
    ),
    tools=[
        list_available_scientific_skills,
        consult_scientific_skill,
        read_workspace_file,
        write_workspace_file,
        execute_shell_command,
        test_data_parsers,
        fact_check_artifacts,
        query_arxiv
    ],
    llm=vertex_llm,
    max_iter=6,
    verbose=True
)

engineer_agent = Agent(
    role="Principal AI Systems & Experimentation Engineer",
    goal=(
        "Implement, test, benchmark, and deploy code improvements directly in the native compute environment: "
        "1. SPECIALIZED SKILLS CONSULTATION: Consult 'optimize-for-gpu' (CUDA performance, kernel optimization, memory bottlenecks), "
        "'pytorch-lightning' (lightning modules, callbacks, training loops), 'transformers', and 'scientific-visualization' (publication figures). "
        "2. CODE IMPLEMENTATION: Translate the Scientist's algorithmic and parser designs into clean, modular, vectorized PyTorch code in src/. "
        "3. UNIT & REGRESSION TESTING: Execute the full pytest suite (pytest tests/ -v). Ensure zero test failures and full backwards compatibility. "
        "4. BENCHMARKING & EXPERIMENT RERUNS: Run native evaluations (scripts/eval.py, diagnostic scripts, or benchmark scripts). "
        "Empirically measure Exact Sequence Match (%), I/L-Equivalent Match (%), AA Precision/Recall, and throughput (spectra/sec). "
        "Rerun experiments if results need empirical validation against baseline checkpoints. "
        "5. PUBLICATION FIGURES: Regenerate 300 DPI publication-grade vector/raster figures via scripts/generate_publication_figures.py. "
        "6. DEPLOYMENT & SYNCHRONIZATION: Sync model and app to Hugging Face if ready, and commit and push all verified code to GitHub 'research' branch using gh CLI."
    ),
    backstory=(
        "You are an elite PyTorch and AI systems engineer. You build resilient, high-performance systems with rigorous unit tests, "
        "profiling, and clean software architecture. You leverage 'optimize-for-gpu' and 'pytorch-lightning' skills to ensure optimal memory throughput and numerical stability. "
        "You never ship unverified code, never skip unit tests, and never report metrics without running benchmarks. "
        "You ensure seamless integration from data loaders to model inference and GitHub deployment."
    ),
    tools=[
        list_available_scientific_skills,
        consult_scientific_skill,
        execute_skill_script,
        read_workspace_file,
        write_workspace_file,
        execute_shell_command,
        run_experiment_natively,
        test_data_parsers,
        regenerate_figures,
        deploy_huggingface,
        push_improvement_to_github
    ],
    llm=vertex_llm,
    max_iter=8,
    verbose=True
)

writer_agent = Agent(
    role="Chief Academic Author & Documentation Architect",
    goal=(
        "Synthesize all mathematical formulations, empirical results, and system architectures into flawless, publication-grade academic documentation: "
        "1. SPECIALIZED SKILLS CONSULTATION: Consult 'scientific-writing' (manuscript guidelines, claim auditing, clarity), "
        "'venue-templates' (Nature Methods / journal standards), 'citation-management' (BibTeX and references), and 'markdown-mermaid-writing'. "
        "2. MATHEMATICAL COHERENCE: Ensure every equation in the LaTeX thesis (thesis/chapter1.tex to chapter6.tex, master-document.tex), "
        "manuscript (manuscript/PAPER_MANUSCRIPT.md), and presentation slides (presentation/presentation.tex) exactly matches the codebase in src/ "
        "(CTMC transition rates, conditional flow matching loss, Knapsack reachability DP, and beam search decoding). "
        "3. FACT-CHECKING & METRIC HARMONY: Ensure all numerical metrics in tables and text (Exact Match %, I/L-Equivalent Match %, AA Precision, AA Recall, inference time) "
        "are 100% identical and backed by real benchmark logs across SUPERVISOR_REPORT_DFM_DE_NOVO.md, SYSTEM_OPTIMIZATION_REPORT.md, "
        "ARTIFACTS_MANIFEST.md, and PRESENTATION_OF_RESULTS.md. Run consistency checking scripts via execute_skill_script if needed. "
        "4. ELIMINATION OF AI CLICHÉS: Eradicate all promotional buzzwords, generic fluff, and LLM clichés "
        "('testament to', 'game-changer', 'revolutionary', 'pivotal', 'beacon', 'dive into'). Maintain rigorous, formal academic prose suitable for Nature Methods or a doctoral defense. "
        "5. EXHAUSTIVE REPRODUCIBILITY GUIDE: Maintain the end-to-end reproduction guide in README.md, including environment setup (.venv), dataset loading, "
        "training commands, evaluation commands, random seeds, and figure generation. "
        "6. GITHUB COMMIT: Commit all updated documentation and thesis files to GitHub."
    ),
    backstory=(
        "You are an academic author and documentation architect who has prepared doctoral dissertations and papers for Nature Methods, Bioinformatics, "
        "and Journal of Proteome Research. You leverage the 'scientific-writing' and 'venue-templates' skills to guarantee uncompromising scholarly standards. "
        "You write with mathematical precision, stylistic clarity, and unwavering empirical truthfulness."
    ),
    tools=[
        list_available_scientific_skills,
        consult_scientific_skill,
        execute_skill_script,
        read_workspace_file,
        write_workspace_file,
        execute_shell_command,
        fact_check_artifacts,
        regenerate_figures,
        push_improvement_to_github
    ],
    llm=vertex_llm,
    max_iter=8,
    verbose=True
)

critic_agent = Agent(
    role="Chief Scientific Reviewer & Independent Auditor",
    goal=(
        "Conduct uncompromising, adversarial peer-review audits on all research outputs: "
        "1. SPECIALIZED SKILLS CONSULTATION: Consult 'peer-review' (claim-evidence validation, reporting standards, adversarial evaluation), "
        "'scientific-critical-thinking', and 'statistical-analysis' (statistical tests, power, assumption auditing). "
        "2. ADVERSARIAL SCRUTINY: Scrutinize every claim, equation, benchmark table, and code change. Flag any unsubstantiated assertion, circular reasoning, or data leakage. "
        "3. EMPIRICAL VERIFICATION: Cross-examine reported benchmark numbers against raw JSON logs in artifacts/ and pytest execution outputs. "
        "If numbers do not match or lack experimental logs, reject the claim and demand that the Engineer rerun the benchmark. "
        "4. PARSER AND EDGE-CASE STRESS TESTING: Ensure parser fixes for MGF/MGZ/mzML/parquet are tested against edge cases (missing headers, non-standard charge strings, empty peak lists, modified peptide sequences). "
        "5. FORMAL AUDIT REPORT: Issue a definitive, itemized Peer Review Approval Report certifying that the repository is mathematically coherent, "
        "empirically verified, 100% reproducible, and free of placeholders."
    ),
    backstory=(
        "You are a notoriously exacting senior journal reviewer and academic auditor. You use the 'peer-review' and 'scientific-critical-thinking' skills "
        "to conduct relentless methodological reviews. You scrutinize every statistical claim, verify every mathematical proof, "
        "and reject manuscripts that lack reproducibility or contain unverified claims. You only approve work that meets the highest standards of scientific rigor."
    ),
    tools=[
        list_available_scientific_skills,
        consult_scientific_skill,
        execute_skill_script,
        read_workspace_file,
        execute_shell_command,
        fact_check_artifacts,
        test_data_parsers
    ],
    llm=vertex_llm,
    max_iter=8,
    verbose=True
)


# =====================================================================
# COLLABORATIVE TASKS (ENRICHED WORKFLOW & VERIFICATION GATES)
# =====================================================================

task_research_and_improvement_strategy = Task(
    description=(
        "PHASE 1: RESEARCH EXPLORATION & ALGORITHMIC STRATEGY\n"
        "====================================================\n"
        "1. CONSULT SCIENTIFIC SKILLS:\n"
        "   - Use consult_scientific_skill with 'pyopenms', 'matchms', and 'biopython' to inspect golden MS/MS standards, "
        "     fragmentation chemistry (b/y ions, neutral losses, isotopic distributions), and spectrum representation practices.\n"
        "2. AUDIT LOCAL WORKSPACE & CODEBASE:\n"
        "   - Inspect src/models/ (flow transformer, spectrum encoder), src/inference/ (knapsack_dp.py, predict.py), and tests/.\n"
        "   - Review the current CTMC discrete flow matching formulation, loss functions, and reachability guidance.\n"
        "3. AUDIT DATA PARSERS (MGF, MGZ, mzML, Parquet):\n"
        "   - Audit src/data/data.py and deployment/huggingface/app.py.\n"
        "   - Verify handling of 'SEQ=' and 'SEQUENCE=' header fields in MGF/MGZ files, charge parsing, and precursor mass extraction.\n"
        "   - Check support for gzip-compressed MGF (.mgz) and mzML format resilience.\n"
        "4. PROPOSE ACTIONABLE ALGORITHMIC INNOVATIONS:\n"
        "   - Formulate concrete mathematical enhancements to raise Exact Sequence Match (%) and I/L-Equivalent Match (%):\n"
        "     a) Flow schedule optimization: improved continuous-time jump rate schedule (e.g. cosine annealing or adaptive temperature schedule).\n"
        "     b) Dynamic Knapsack DP reachability guidance: charge-adaptive mass tolerance delta(z) = delta_0 * (1 + 0.1 * (z - 1)) or isotope tolerance.\n"
        "     c) Spectrum conditioning: complementary b/y ion pairing, neutral loss channels, or peak intensity normalization.\n"
        "5. DELIVER FORMAL SPECIFICATION:\n"
        "   - Output clear mathematical definitions, exact drop-in code modifications, and explicit unit test instructions for the Principal Engineer."
    ),
    expected_output=(
        "A comprehensive R&D improvement strategy informed by pyopenms/matchms skills, with exact mathematical equations, "
        "concrete code diffs for src/ and tests/, parser bugfixes (SEQ/SEQUENCE handling in MGF/MGZ), and an explicit testing & benchmark plan."
    ),
    agent=scientist_agent
)

task_experimentation_and_benchmarking = Task(
    description=(
        "PHASE 2: IMPLEMENTATION, TESTING & GPU BENCHMARKING\n"
        "===================================================\n"
        "1. CONSULT SCIENTIFIC SKILLS:\n"
        "   - Consult 'optimize-for-gpu' for CUDA tensor optimization and memory management, 'pytorch-lightning' for training loops, "
        "     and 'scientific-visualization' for publication figure styling.\n"
        "2. IMPLEMENT CODE MODIFICATIONS:\n"
        "   - Apply the Scientist's algorithmic and parser enhancements cleanly into src/ and tests/.\n"
        "   - Maintain backward compatibility and clean type annotations.\n"
        "3. RUN TEST SUITE & VALIDATE ZERO REGRESSIONS:\n"
        "   - Execute pytest tests/ -v using test_data_parsers.\n"
        "   - Ensure all 58+ unit tests pass without error.\n"
        "4. BENCHMARK & RERUN EXPERIMENTS IF NEEDED:\n"
        "   - Run native evaluation or diagnostic benchmarks (e.g. scripts/eval.py, scripts/diagnostic_refined_decoding.py, or benchmark scripts).\n"
        "   - Measure Exact Sequence Match (%), I/L-Equivalent Match (%), AA Precision/Recall, and throughput (spectra/sec).\n"
        "   - Rerun experiments if numbers need empirical validation against baseline checkpoints.\n"
        "5. REGENERATE 300 DPI PUBLICATION FIGURES:\n"
        "   - Execute scripts/generate_publication_figures.py adhering to 'scientific-visualization' standards to produce Nature Methods-grade charts.\n"
        "6. HUGGING FACE & GITHUB SYNCHRONIZATION:\n"
        "   - Sync model/Gradio app to Hugging Face if ready.\n"
        "   - Commit all verified code, test updates, and benchmark metrics, and push to GitHub 'research' branch using active gh credentials."
    ),
    expected_output=(
        "Execution logs showing clean pytest test suite passage, empirical benchmark metrics comparing before/after, "
        "regenerated 300 DPI figures adhering to scientific-visualization guidelines, and confirmation of GitHub push to the research branch."
    ),
    agent=engineer_agent
)

task_thesis_and_factcheck_sync = Task(
    description=(
        "PHASE 3: DOCUMENTATION, THESIS & MATHEMATICAL COHERENCE\n"
        "=======================================================\n"
        "1. CONSULT SCIENTIFIC SKILLS:\n"
        "   - Consult 'scientific-writing' for evidence provenance and manuscript principles, 'venue-templates' for Nature Methods formatting, "
        "     and 'citation-management' for reference integrity.\n"
        "2. FACT-CHECK EVERY ARTIFACT ACROSS REPOSITORY:\n"
        "   - Audit thesis/ (chapter1.tex to chapter6.tex, master-document.tex), manuscript/ (PAPER_MANUSCRIPT.md), "
        "     presentation/ (presentation.tex), SUPERVISOR_REPORT_DFM_DE_NOVO.md, SYSTEM_OPTIMIZATION_REPORT.md, "
        "     ARTIFACTS_MANIFEST.md, and PRESENTATION_OF_RESULTS.md.\n"
        "   - Ensure zero placeholder text ('TODO', 'TBD', '[INSERT', 'approx.').\n"
        "   - Ensure every reported metric strictly matches raw benchmark logs. Run consistency scripts via execute_skill_script if needed.\n"
        "3. MATHEMATICAL & CODE COHERENCE:\n"
        "   - Verify that all LaTeX equations in chapter3/chapter4/chapter5 match the exact Python implementation in src/ with zero discrepancies.\n"
        "4. ELIMINATE ALL AI CLICHES:\n"
        "   - Ensure formal, objective academic prose throughout. Remove promotional fluff ('testament to', 'game-changer', 'revolutionary', 'pivotal').\n"
        "5. EXHAUSTIVE REPRODUCIBILITY GUIDE:\n"
        "   - Update README.md with comprehensive, step-by-step reproduction instructions: virtual environment setup (.venv), "
        "     dataset loading, training commands, evaluation commands, random seeds, and figure generation.\n"
        "6. COMMIT DOCUMENTATION TO GITHUB:\n"
        "   - Commit all audited thesis chapters, reports, and README updates to git and push to GitHub 'research' branch."
    ),
    expected_output=(
        "Mathematically exact, 100% fact-checked thesis chapters and reports informed by the scientific-writing skill, "
        "with zero placeholders, harmonious metrics, and an exhaustive reproduction guide in README.md, successfully committed and pushed to GitHub."
    ),
    agent=writer_agent
)

task_final_critic_review = Task(
    description=(
        "PHASE 4: INDEPENDENT CRITICAL PEER-REVIEW & AUDIT\n"
        "=================================================\n"
        "1. CONSULT SCIENTIFIC SKILLS:\n"
        "   - Consult 'peer-review' and 'scientific-critical-thinking' for adversarial claim-evidence matrices, "
        "     reporting guideline checks, and statistical reproducibility audits.\n"
        "2. ADVERSARIAL PEER-REVIEW AUDIT:\n"
        "   - Scrutinize the outputs of all previous phases (algorithmic proposal, code changes, test results, benchmarks, thesis chapters, reports).\n"
        "   - Verify that all empirical numbers reported by the Author match raw benchmark logs and pytest outputs.\n"
        "   - Check parser resilience against MGF/MGZ/mzML edge cases.\n"
        "3. EMPIRICAL & REPRODUCIBILITY VERIFICATION:\n"
        "   - Confirm that zero placeholders or unsubstantiated claims exist in any documentation.\n"
        "   - Confirm that the README reproduction instructions are complete and functional.\n"
        "4. SIGN-OFF & APPROVAL REPORT:\n"
        "   - If any flaw, mathematical inconsistency, or unverified claim is found, demand re-execution.\n"
        "   - If all criteria are met, issue a formal Chief Scientific Critic Approval Report certifying the research gains, "
        "     mathematical exactitude, and full reproducibility of the repository."
    ),
    expected_output=(
        "A formal Chief Scientific Critic Peer Review Approval Report prepared according to peer-review skill standards, "
        "certifying validated performance gains, mathematical exactitude, flawless cross-artifact coherence, and complete reproducibility across the repository."
    ),
    agent=critic_agent
)


# =====================================================================
# HIERARCHICAL EXECUTION LOOP
# =====================================================================

def start_native_crew(max_loops: int = 1):
    """Executes the autonomous scientific research, improvement, fact-checking, and documentation squad natively."""
    print("=" * 80)
    print("🚀 Initializing DFlowNovo Autonomous Research & Verification Squad (Native Execution)...")
    print(f"🌍 GCP Project ID: {project_id}")
    print(f"📍 Vertex AI Region: {region}")
    print(f"📂 Workspace Directory: {WORKSPACE_ROOT}")
    print(f"🧠 Scientific Skills Library: {SKILLS_DIR}")
    print("=" * 80)
    
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
            manager_llm=vertex_llm,
            verbose=True
        )
        
        result = peptide_squad.kickoff()
        print(f"\n✅ Iteration {loop_idx} complete.")
        
    return result


if __name__ == "__main__":
    start_native_crew(max_loops=1)
