import os
import requests
import sys

# GitLab environment variables
GITLAB_API_URL = os.getenv('CI_API_V4_URL')
PROJECT_ID = os.getenv('CI_PROJECT_ID')
MR_IID = os.getenv('CI_MERGE_REQUEST_IID')  # Merge request internal ID
PRIVATE_TOKEN = os.getenv('GITLAB_PRIVATE_TOKEN')  # GitLab API token

# Set the maximum allowed lines of change
MAX_LINES = 500

def get_mr_changes():
    """Fetch merge request changes from GitLab API."""
    headers = {
        'PRIVATE-TOKEN': PRIVATE_TOKEN
    }

    # Get the MR changes (additions, deletions, etc.)
    url = f"{GITLAB_API_URL}/projects/{PROJECT_ID}/merge_requests/{MR_IID}/changes"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch MR changes: {response.status_code}")
        sys.exit(1)

    changes = response.json()

    # Count the total number of additions (+) and deletions (-) in the diff
    total_additions = 0
    total_deletions = 0

    for file in changes['changes']:
        diff = file.get('diff', '')
        additions = sum(1 for line in diff.splitlines() if line.startswith('+') and not line.startswith('+++'))
        deletions = sum(1 for line in diff.splitlines() if line.startswith('-') and not line.startswith('---'))

        total_additions += additions
        total_deletions += deletions

    total_changes = total_additions + total_deletions

    return total_changes

if __name__ == '__main__':
    total_lines = get_mr_changes()
    
    print(f"Total lines changed: {total_lines}")

    if total_lines > MAX_LINES:
        print(f"Error: MR exceeds {MAX_LINES} lines of changes ({total_lines} lines).")
        sys.exit(1)
    else:
        print(f"MR is within the limit of {MAX_LINES} lines ({total_lines} lines).")
        sys.exit(0)