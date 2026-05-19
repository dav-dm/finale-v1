import subprocess

def get_git_commit_sha():
    """
    Returns the SHA of the current git commit for the repository
    where this code is being executed.
    Returns an empty string if not in a git repository or git is not available.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            universal_newlines=True
        ).strip()
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""