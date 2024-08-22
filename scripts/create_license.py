import argparse
import os
import pathlib
import subprocess
from datetime import datetime, timedelta

import jwt

_current_file = pathlib.Path(__file__)
SECRETS_DIR_PATH = _current_file.parent / pathlib.Path("../../secrets")

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Generate an expiring license using the Langsmith private key"
)
parser.add_argument(
    "-e",
    "--expiration-weeks",
    type=int,
    default=1,
    help="JWT expiration time in weeks (default: 1 week)",
)
parser.add_argument(
    "-c",
    "--customer-name",
    type=str,
    default="dev",
    help="Name of the customer license is issued for (default: dev)",
)
args = parser.parse_args()

# Load the private key from an environment variable. This is in the Langchain 1password vault.
private_key_str = os.environ.get("LANGGRAPH_CLOUD_LICENSE_PRIVATE_KEY")

if private_key_str is None:
    print(
        "LANGGRAPH_CLOUD_LICENSE_PRIVATE_KEY environment variable not found. Please set it."
    )
    # Run the 1Password CLI command to fetch the item details
    command = "op read 'op://Engineering/LangGraph Cloud License Keys/Private'"
    try:
        private_key_str = subprocess.check_output(
            command, shell=True, universal_newlines=True
        )
        print("Setting LANGGRAPH_CLOUD_LICENSE_PRIVATE_KEY environment variable")
        os.environ["LANGGRAPH_CLOUD_LICENSE_PRIVATE_KEY"] = private_key_str
    except Exception as e:
        print(f"Error fetching private key from 1Password: {e}")
# Define payload data for your JWT
payload = {
    "sub": args.customer_name,
    "iat": datetime.utcnow(),
    "exp": datetime.utcnow() + timedelta(weeks=args.expiration_weeks),
    "aud": "langgraph-cloud",
}

# Generate the JWT using the loaded private key
jwt_token = jwt.encode(payload, private_key_str, algorithm="RS256")

print(f"JWT Token: {jwt_token}")
