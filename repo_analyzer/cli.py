import argparse

def main():
    # New argument parse.
    # Helpful message thats shown when the user runs your script
    parse = argparse.ArgumentParser(
        description="Generate a tutorial for a GitHub codebase or local directory.")
    
    # Can only supply one of the arguments. [--repo, --dir]
    source_group = parse.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--repo", help="URL of the GitHub repository (e.g., 'https://github.com/user/repo')")
    source_group.add_argument("--dir", help="Path to local directory.")

    parse.add_argument("-n", "--name", help="Project name(optional, derived from repo/directory if omitted).")

    args = parse.parse_args()    
