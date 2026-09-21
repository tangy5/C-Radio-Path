import argparse
from jira import JIRA

def create_jira_issue(jira_server, username, api_token, project_key, issue_type, summary, description, priority):
    try:
        jira = JIRA(server=jira_server, token_auth=api_token)

        issue_dict = {
            'project': {'key': project_key},
            'summary': summary,
            'description': description,
            'issuetype': {'name': issue_type},
            'priority': {'name': priority}
        }

        new_issue = jira.create_issue(fields=issue_dict)
        print(f"Successfully created Jira issue: {new_issue.key}")
        jira.assign_issue(issue=new_issue, assignee=username)
        print(f"Issue {new_issue.key} is now assigned to {username}")
        print(f"Link to JIRA ticket: https://jirasw.nvidia.com/browse/{new_issue.key}")
    except Exception as e:
        print(f"Failed to create Jira issue: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Jira issue via command line.")
    parser.add_argument("--jira_server", required=True, help="URL of the Jira server")
    parser.add_argument("--username", required=True, help="Jira username or assignee")
    parser.add_argument("--api_token", required=True, help="API token for Jira authentication")
    parser.add_argument("--project_key", required=True, help="Project key (e.g., TLT)")
    parser.add_argument("--issue_type", default="Bug", help="Type of the issue (e.g., Bug, Task)")
    parser.add_argument("--summary", required=True, help="Summary of the issue")
    parser.add_argument("--description", required=True, help="Detailed description of the issue")
    parser.add_argument("--priority", default="P1 - Should have", help="Priority of the issue")

    args = parser.parse_args()

    create_jira_issue(
        jira_server=args.jira_server,
        username=args.username,
        api_token=args.api_token,
        project_key=args.project_key,
        issue_type=args.issue_type,
        summary=args.summary,
        description=args.description,
        priority=args.priority
    )


"""
Usage:
python create_jira_issue.py \
  --jira_server <jira-server> \
  --username <user-name> \
  --api_token <api-token> \
  --project_key <project-key> \
  --summary <summary> \
  --description <description>
"""