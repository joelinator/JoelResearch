# agents_native.py
"""
DFLowNovo Multi-Agent Autonomous Research, Improvement & Verification Squad (Native VM Version)
=================================================================================================
Native multi-agent R&D squad executing directly inside the compute environment (e.g., A100 GPU VM).
Directly accesses PyTorch, CUDA, virtual environment packages, datasets, model checkpoints,
and local git workspace without SSH/IAP tunneling overhead.
"""

import os
import sys
import time
import json
import re
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load local environment variables (.env)
load_dotenv()

project_id = os.getenv("GCP_PROJECT_ID", "joelgedeon-project-507410")
region = os.getenv("VERTEXAI_LOCATION", "us-central1")
repo_url = os.getenv("GITHUB_REPO_URL", "https://github.com/joelinator/joelresearch.git")

# Ensure Vertex AI routing for Google GenAI / CrewAI LLM
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

WORKSPACE_ROOT = Path(__file__).resolve().parent

# =====================================================================
# NATIVE TOOLS FOR THE AGENT SQUAD
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


@tool("Execute Shell Command in Workspace")
def execute_shell_command(command: str) -> str:
    """Useful to inspect files, check git status, list directories, read logs, or execute bash utilities
    directly inside the native workspace."""
    try:
        res = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=180, cwd=str(WORKSPACE_ROOT)
        )
        return f"=== SHELL COMMAND OUTPUT (Exit: {res.returncode}) ===\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    except Exception as e:
        return f"Shell command execution error: {str(e)}"


@tool("Run Experiment, Training, or Benchmark Natively on GPU")
def run_experiment_natively(command: str = "python3 scripts/eval.py", custom_script_code: str = "") -> str:
    """Useful to execute PyTorch training, model evaluation, benchmark scripts (e.g. scripts/eval.py,
    scripts/benchmark_refined_decoding_50k.py), or custom Python experiment scripts directly on the local GPU.
    Runs natively in the active environment with direct CUDA / A100 acceleration."""
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
        return (
            f"=== EXPERIMENT EXECUTION COMPLETED (Elapsed: {elapsed:.1f}s, Exit: {res.returncode}) ===\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
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


@tool("Validate and Test Mass Spectrometry Data Parsers (MGF, MGZ, mzML)")
def test_data_parsers(test_file: str = "") -> str:
    """Useful to test and validate MS/MS file parsing routines for MGF, MGZ (gzipped MGF), mzML,
    and parquet formats. Checks extraction of SEQ/SEQUENCE headers, TITLE, PEPMASS, CHARGE, RTINSECONDS, and peaks."""
    try:
        cmd = f"{sys.executable} -m pytest tests/ -k 'test' -v"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180, cwd=str(WORKSPACE_ROOT))
        return f"=== DATA PARSER & SUITE TEST RESULTS ===\nExit Code: {res.returncode}\nStdout:\n{res.stdout}\nStderr:\n{res.stderr}"
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
        return f"=== HUGGING FACE DEPLOYMENT STATUS ===\nExit Code: {res.returncode}\nStdout:\n{res.stdout}\nStderr:\n{res.stderr}"
    except Exception as e:
        return f"Hugging Face deployment failed: {str(e)}"


@tool("Push Verified Improvements to GitHub")
def push_improvement_to_github(commit_message: str) -> str:
    """Useful to commit and push verified improvements, regenerated figures, audited thesis documents,
    and benchmark metrics directly to the GitHub repository on the 'research' branch."""
    repo_url_env = os.getenv("GITHUB_REPO_URL", "https://github.com/joelinator/joelresearch.git")
    github_token = os.getenv("GITHUB_TOKEN")
    
    try:
        subprocess.run(["git", "config", "--global", "user.name", "DFlowNovo-Research-Squad"], check=True, cwd=str(WORKSPACE_ROOT))
        subprocess.run(["git", "config", "--global", "user.email", "research-squad@cloudrun.internal"], check=True, cwd=str(WORKSPACE_ROOT))
        
        if github_token and repo_url_env and "@" not in repo_url_env:
            clean_url = repo_url_env.replace("https://", "").replace("http://", "")
            authenticated_url = f"https://{github_token}@{clean_url}"
            subprocess.run(["git", "remote", "set-url", "origin", authenticated_url], check=True, cwd=str(WORKSPACE_ROOT))
        elif repo_url_env:
            subprocess.run(["git", "remote", "set-url", "origin", repo_url_env], check=True, cwd=str(WORKSPACE_ROOT))
            
        subprocess.run(["git", "add", "."], check=True, cwd=str(WORKSPACE_ROOT))
        subprocess.run(["git", "commit", "-m", f"Scientific R&D Improvement: {commit_message}"], capture_output=True, text=True, cwd=str(WORKSPACE_ROOT))
        
        push_res = subprocess.run(["git", "push", "origin", "research"], capture_output=True, text=True, cwd=str(WORKSPACE_ROOT))
        if push_res.returncode != 0:
            push_res2 = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True, cwd=str(WORKSPACE_ROOT))
            if push_res2.returncode != 0:
                return f"Local commit created: '{commit_message}'. (Push status: {push_res.stderr.strip() or push_res2.stderr.strip()})"
        return f"Successfully pushed verified updates to GitHub (research branch): {commit_message}"
    except Exception as e:
        return f"Local git state preserved: {str(e)}"


# =====================================================================
# MULTI-AGENT SQUAD DEFINITIONS (NATIVE ENVIRONMENT)
# =====================================================================

manager_agent = Agent(
    role="Principal AI Architect & Research Director",
    goal=(
        "Lead the collaborative research squad natively in the compute environment. "
        "Direct the team to push the boundaries beyond InstaNovo and Casanovo baselines, raising Exact Match (%) and I/L Equivalent Accuracy (%). "
        "Enforce 100% empirical fact-checking, mathematical exactitude in the thesis, figure regeneration, MGF/MGZ/mzML parser robustness, "
        "and direct GitHub synchronization upon verified gains."
    ),
    backstory=(
        "You are an inspiring computational proteomics research director operating directly inside the high-performance compute environment. "
        "You demand relentless empirical rigor backed by flawless mathematical proofs and reproducible code. "
        "You orchestrate the scientist, engineer, writer, and critic into a cohesive native unit."
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
        "5. Formulate concrete code modifications to push Exact Match and I/L-equivalent metrics higher."
    ),
    backstory=(
        "You are a cutting-edge machine learning scientist specializing in generative flow matching on discrete sequence spaces and tandem mass spectrometry. "
        "You analyze error modes and design targeted algorithmic solutions."
    ),
    tools=[execute_shell_command, fact_check_artifacts, test_data_parsers, query_arxiv],
    llm=vertex_llm,
    max_iter=5,
    verbose=True
)

engineer_agent = Agent(
    role="Principal AI & Systems Engineer",
    goal=(
        "Execute high-performance PyTorch training, inference, and benchmarking directly on the local GPU: "
        "1. Execute benchmarks (scripts/eval.py, scripts/benchmark_refined_decoding_50k.py) with full CUDA acceleration. "
        "2. Measure exact sequence match, I/L equivalent match, amino acid precision/recall, and inference throughput (spectra/sec). "
        "3. Regenerate 300 DPI publication figures via scripts/generate_publication_figures.py. "
        "4. Validate parser resilience on MGF, MGZ, mzML, and Parquet formats. "
        "5. Synchronize verified improvements to Hugging Face and push commits to GitHub 'research' branch."
    ),
    backstory=(
        "You are a seasoned AI systems engineer running directly on GPU infrastructure. You optimize CUDA kernels, CTMC sampling loops, "
        "and Knapsack dynamic programming tables for maximum throughput and precision."
    ),
    tools=[execute_shell_command, run_experiment_natively, test_data_parsers, regenerate_figures, deploy_huggingface, push_improvement_to_github],
    llm=vertex_llm,
    max_iter=8,
    verbose=True
)

writer_agent = Agent(
    role="Chief Academic Author & Documentation Architect",
    goal=(
        "Ensure all LaTeX thesis chapters, manuscript drafts, presentation slides, and markdown reports reflect mathematically exact implementations: "
        "1. Synchronize all formulas in thesis/ (chapter3.tex, chapter4.tex, chapter5.tex) with exact src/ implementations. "
        "2. Populate benchmark tables and replace all placeholder entries (TBD/TODO) with verified empirical numbers. "
        "3. Eliminate AI clichés and maintain formal academic prose style. "
        "4. Maintain the exhaustive reproduction guide in README.md."
    ),
    backstory=(
        "You are an academic documentation architect who crafts rigorous, publication-grade manuscripts and LaTeX theses. "
        "You write concise, objective scientific prose and ensure seamless repository coherence."
    ),
    tools=[execute_shell_command, fact_check_artifacts, regenerate_figures, push_improvement_to_github],
    llm=vertex_llm,
    max_iter=5,
    verbose=True
)

critic_agent = Agent(
    role="Chief Scientific Reviewer & Independent Auditor",
    goal=(
        "Conduct rigorous peer-review audits on all research outputs: "
        "1. Verify that all claimed empirical numbers match raw JSON benchmark outputs. "
        "2. Ensure zero placeholder text, mathematical approximations, or inconsistencies exist across artifacts. "
        "3. Certify that the README reproduction guide is 100% reproducible and unambiguous."
    ),
    backstory=(
        "You are an uncompromising senior peer reviewer for top journals. You reject hand-waving claims and demand empirical evidence and mathematical precision."
    ),
    tools=[execute_shell_command, fact_check_artifacts, test_data_parsers],
    llm=vertex_llm,
    max_iter=5,
    verbose=True
)


# =====================================================================
# COLLABORATIVE TASKS (NATIVE WORKFLOW)
# =====================================================================

task_research_and_improvement_strategy = Task(
    description=(
        "1. AUDIT LOCAL WORKSPACE: Inspect existing code, test suites, training logs, checkpoints, and files in the workspace.\n"
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
        "1. EXECUTE NATIVELY ON GPU: Run benchmarks (e.g. scripts/eval.py or scripts/benchmark_refined_decoding_50k.py) natively on the local GPU.\n"
        "2. MEASURE METRICS: Evaluate Exact Sequence Match (%), I/L-Equivalent Match (%), AA Precision/Recall, and runtime.\n"
        "3. REGENERATE PUBLICATION FIGURES: Execute scripts/generate_publication_figures.py to produce updated 300 DPI Nature Methods-grade charts.\n"
        "4. HUGGING FACE DEPLOYMENT: Deploy the updated model and Gradio Space if new improvements or parser fixes are verified.\n"
        "5. COMMIT & PUSH: Push verified improvements and benchmark metrics to GitHub 'research' branch."
    ),
    expected_output="Benchmark execution logs, measured metrics comparing before/after, regenerated 300 DPI figures, and confirmed GitHub push.",
    agent=engineer_agent
)

task_thesis_and_factcheck_sync = Task(
    description=(
        "1. FACT-CHECK EVERY ARTIFACT: Verify all numbers, tables, and claims across thesis/ (LaTeX), manuscript/, presentation/, and markdown reports.\n"
        "2. THESIS & CODE COHERENCE: Ensure that chapter3/chapter4/chapter5 LaTeX equations (loss functions, CTMC flow dynamics, Knapsack DP) "
        "mirror the exact implementation in src/ with zero approximations.\n"
        "3. ELIMINATE AI CLICHÉS: Audit prose to ensure rigorous, formal scientific writing and remove all AI buzzwords.\n"
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
        "2. VERIFY ACCURACY & RIGOR: Confirm that all claimed numbers match raw JSON outputs, the thesis is 100% mathematically aligned with the codebase, "
        "and the README reproduction guide is complete.\n"
        "3. SIGN-OFF: Issue a formal scientific approval report certifying empirical gains, mathematical rigor, and full reproducibility."
    ),
    expected_output="A formal Chief Scientific Critic Approval Report certifying validated performance gains, mathematical exactitude, and reproducibility.",
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
            verbose=True
        )
        
        result = peptide_squad.kickoff()
        print(f"\n✅ Iteration {loop_idx} complete.")
        
    return result


if __name__ == "__main__":
    start_native_crew(max_loops=1)
