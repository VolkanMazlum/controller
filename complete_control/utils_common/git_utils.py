import subprocess
from pathlib import Path


def get_git_commit_hash(repo_path: str | Path) -> str:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return process.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Log error or handle as appropriate
        print(f"Error getting git hash for {repo_path}: {e}")
        return "unknown"
    except FileNotFoundError:
        print(f"Git command not found. Ensure git is installed and in PATH.")
        return "git_not_found"
