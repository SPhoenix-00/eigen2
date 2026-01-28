import os
import json
import time
import urllib.request
import sys

def log_to_debug(message, data):
    payload = {
        "timestamp": int(time.time() * 1000),
        "location": "debug_env_check.py",
        "message": message,
        "data": data,
        "sessionId": "debug-session",
        "runId": "env_check",
        "hypothesisId": "env_vars_missing"
    }
    
    # Write to local file
    log_path = r"d:\GitHub\eigen2\.cursor\debug.log"
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as e:
        print(f"Failed to write to log file: {e}")

    # Try HTTP
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:7242/ingest/c1d62411-6c1d-478b-9120-90d035db21a9", 
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        urllib.request.urlopen(req, timeout=1)
    except:
        pass

vars_to_check = ["CLOUD_PROVIDER", "CLOUD_BUCKET", "GOOGLE_APPLICATION_CREDENTIALS"]
data = {v: os.environ.get(v, "MISSING") for v in vars_to_check}

# Check for credential file in CWD
local_creds = "gcs-credentials.json"
data["local_creds_exists"] = os.path.exists(local_creds)
data["cwd"] = os.getcwd()
data["platform"] = sys.platform

# Check if the env var path exists if set
if data["GOOGLE_APPLICATION_CREDENTIALS"] != "MISSING":
    data["env_creds_path_exists"] = os.path.exists(data["GOOGLE_APPLICATION_CREDENTIALS"])

print(f"Environment Check: {json.dumps(data, indent=2)}")
log_to_debug("Environment Variable Check", data)

