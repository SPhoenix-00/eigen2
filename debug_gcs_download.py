"""
Diagnostic script to debug GCS download issues.
Lists bucket contents and tests multiple file paths.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Determine log path - works on both Windows and Linux
if sys.platform == 'win32':
    log_path = r"d:\GitHub\eigen2\.cursor\debug.log"
else:
    # For Linux/remote systems, use workspace root
    workspace_root = os.environ.get('WORKSPACE', '/workspace')
    if os.path.exists('/workspace'):
        log_path = '/workspace/.cursor/debug.log'
    else:
        log_path = os.path.join(os.getcwd(), '.cursor', 'debug.log')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

# #region agent log
with open(log_path, "a") as f:
    f.write(json.dumps({
        "id": f"log_{int(datetime.now().timestamp() * 1000)}",
        "timestamp": int(datetime.now().timestamp() * 1000),
        "location": "debug_gcs_download.py:12",
        "message": "Script started",
        "data": {"hypothesisId": "A"},
        "sessionId": "debug-session",
        "runId": "run1"
    }) + "\n")
# #endregion

try:
    from google.cloud import storage
    
    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "id": f"log_{int(datetime.now().timestamp() * 1000)}",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "location": "debug_gcs_download.py:25",
            "message": "GCS library imported successfully",
            "data": {"hypothesisId": "A"},
            "sessionId": "debug-session",
            "runId": "run1"
        }) + "\n")
    # #endregion
    
    # Initialize client
    bucket_name = 'eigen2-checkpoints-ase0'
    
    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "id": f"log_{int(datetime.now().timestamp() * 1000)}",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "location": "debug_gcs_download.py:40",
            "message": "Initializing GCS client",
            "data": {"bucket_name": bucket_name, "hypothesisId": "A"},
            "sessionId": "debug-session",
            "runId": "run1"
        }) + "\n")
    # #endregion
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "id": f"log_{int(datetime.now().timestamp() * 1000)}",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "location": "debug_gcs_download.py:50",
            "message": "GCS client initialized",
            "data": {"hypothesisId": "A"},
            "sessionId": "debug-session",
            "runId": "run1"
        }) + "\n")
    # #endregion
    
    print(f"\n{'='*70}")
    print("GCS Download Diagnostic")
    print(f"{'='*70}\n")
    print(f"Bucket: {bucket_name}\n")
    
    # Hypothesis A: File path is wrong (missing eigen2/ prefix)
    # Hypothesis B: File doesn't exist at all
    # Hypothesis C: File exists with different name/date
    # Hypothesis D: Wrong bucket name
    # Hypothesis E: Credentials issue
    
    # Test 1: List all files with Eigen2_Master in name
    print("Searching for files matching 'Eigen2_Master'...")
    
    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "id": f"log_{int(datetime.now().timestamp() * 1000)}",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "location": "debug_gcs_download.py:70",
            "message": "Starting file search",
            "data": {"search_pattern": "Eigen2_Master", "hypothesisId": "B"},
            "sessionId": "debug-session",
            "runId": "run1"
        }) + "\n")
    # #endregion
    
    matching_files = []
    all_files = []
    
    try:
        blobs = bucket.list_blobs()
        for blob in blobs:
            all_files.append(blob.name)
            if 'Eigen2_Master' in blob.name:
                matching_files.append(blob.name)
    except Exception as e:
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "id": f"log_{int(datetime.now().timestamp() * 1000)}",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "location": "debug_gcs_download.py:90",
                "message": "Error listing blobs",
                "data": {"error": str(e), "hypothesisId": "E"},
                "sessionId": "debug-session",
                "runId": "run1"
            }) + "\n")
        # #endregion
        print(f"✗ Error listing bucket contents: {e}")
        matching_files = []
    
    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "id": f"log_{int(datetime.now().timestamp() * 1000)}",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "location": "debug_gcs_download.py:100",
            "message": "File search completed",
            "data": {
                "total_files": len(all_files),
                "matching_files": matching_files,
                "matching_count": len(matching_files),
                "hypothesisId": "B"
            },
            "sessionId": "debug-session",
            "runId": "run1"
        }) + "\n")
    # #endregion
    
    if matching_files:
        print(f"✓ Found {len(matching_files)} matching file(s):")
        for fname in matching_files:
            print(f"  - {fname}")
    else:
        print("✗ No files matching 'Eigen2_Master' found")
    
    print(f"\nTotal files in bucket: {len(all_files)}")
    if len(all_files) > 0 and len(all_files) <= 20:
        print("\nAll files in bucket:")
        for fname in sorted(all_files):
            print(f"  - {fname}")
    elif len(all_files) > 20:
        print(f"\nFirst 20 files in bucket:")
        for fname in sorted(all_files)[:20]:
            print(f"  - {fname}")
    
    # Test 2: Try multiple possible paths
    print(f"\n{'='*70}")
    print("Testing possible file paths...")
    print(f"{'='*70}\n")
    
    test_paths = [
        'Eigen2_Master_PY_OUTPUT_311225.pkl',  # Original (wrong)
        'eigen2/Eigen2_Master_PY_OUTPUT_311225.pkl',  # With eigen2/ prefix
        'Eigen2_Master_PY_OUTPUT.pkl',  # Without date suffix
        'eigen2/Eigen2_Master_PY_OUTPUT.pkl',  # Without date, with prefix
    ]
    
    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "id": f"log_{int(datetime.now().timestamp() * 1000)}",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "location": "debug_gcs_download.py:145",
            "message": "Starting path tests",
            "data": {"test_paths": test_paths, "hypothesisId": "A"},
            "sessionId": "debug-session",
            "runId": "run1"
        }) + "\n")
    # #endregion
    
    for test_path in test_paths:
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "id": f"log_{int(datetime.now().timestamp() * 1000)}",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "location": "debug_gcs_download.py:155",
                "message": "Testing path",
                "data": {"test_path": test_path, "hypothesisId": "A"},
                "sessionId": "debug-session",
                "runId": "run1"
            }) + "\n")
        # #endregion
        
        blob = bucket.blob(test_path)
        
        # Test 1: Check if exists() works
        try:
            exists = blob.exists()
        except Exception as e:
            exists = None
            # #region agent log
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "id": f"log_{int(datetime.now().timestamp() * 1000)}",
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "location": "debug_gcs_download.py:170",
                    "message": "exists() check failed",
                    "data": {
                        "test_path": test_path,
                        "error": str(e),
                        "hypothesisId": "E"
                    },
                    "sessionId": "debug-session",
                    "runId": "run1"
                }) + "\n")
            # #endregion
        
        # Test 2: Try actual download (even if exists() fails)
        download_success = False
        download_error = None
        test_file = f"/tmp/test_download_{test_path.replace('/', '_')}"
        
        try:
            # #region agent log
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "id": f"log_{int(datetime.now().timestamp() * 1000)}",
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "location": "debug_gcs_download.py:200",
                    "message": "Attempting download test",
                    "data": {"test_path": test_path, "hypothesisId": "A"},
                    "sessionId": "debug-session",
                    "runId": "run1"
                }) + "\n")
            # #endregion
            
            blob.download_to_filename(test_file, timeout=10)
            download_success = True
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)
        except Exception as e:
            download_error = str(e)
            if os.path.exists(test_file):
                os.remove(test_file)
        
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "id": f"log_{int(datetime.now().timestamp() * 1000)}",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "location": "debug_gcs_download.py:225",
                "message": "Path test result",
                "data": {
                    "test_path": test_path,
                    "exists_check": exists,
                    "download_success": download_success,
                    "download_error": download_error,
                    "hypothesisId": "A"
                },
                "sessionId": "debug-session",
                "runId": "run1"
            }) + "\n")
        # #endregion
        
        if download_success:
            status = "✓ DOWNLOADABLE"
        elif exists is True:
            status = "✓ EXISTS (download not tested)"
        elif exists is False:
            status = "✗ NOT FOUND"
        else:
            status = "? EXISTS CHECK FAILED"
        
        print(f"{status}: {test_path}")
        if download_error and not download_success:
            print(f"    Error: {download_error[:100]}")
    
    # Test 3: Detailed test of the 311225 file specifically
    print(f"\n{'='*70}")
    print("Detailed Test: eigen2/Eigen2_Master_PY_OUTPUT_311225.pkl")
    print(f"{'='*70}\n")
    
    target_path = 'eigen2/Eigen2_Master_PY_OUTPUT_311225.pkl'
    blob = bucket.blob(target_path)
    
    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "id": f"log_{int(datetime.now().timestamp() * 1000)}",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "location": "debug_gcs_download.py:240",
            "message": "Testing target file specifically",
            "data": {"target_path": target_path, "hypothesisId": "E"},
            "sessionId": "debug-session",
            "runId": "run1"
        }) + "\n")
    # #endregion
    
    print(f"Target file: {target_path}")
    print(f"Full GCS path: gs://{bucket_name}/{target_path}\n")
    
    # Try exists check
    try:
        exists = blob.exists()
        print(f"blob.exists(): {exists}")
    except Exception as e:
        print(f"blob.exists() error: {e}")
        exists = None
    
    # Try to get metadata
    try:
        blob.reload()
        print(f"✓ File metadata accessible")
        print(f"  Size: {blob.size} bytes")
        print(f"  Created: {blob.time_created}")
        print(f"  Content-Type: {blob.content_type}")
    except Exception as e:
        print(f"✗ Cannot access file metadata: {e}")
        print(f"  This suggests a permissions issue.")
        print(f"  The file may exist but the service account lacks 'storage.objects.get' permission.")
    
    # Try actual download
    print(f"\nAttempting download test...")
    test_file = "/tmp/test_311225_download.pkl"
    try:
        blob.download_to_filename(test_file, timeout=30)
        file_size = os.path.getsize(test_file)
        print(f"✓ Download successful! ({file_size} bytes)")
        os.remove(test_file)
    except Exception as e:
        print(f"✗ Download failed: {e}")
        if "404" in str(e) or "NotFound" in str(e):
            print(f"\n  Possible causes:")
            print(f"  1. File doesn't exist at this exact path")
            print(f"  2. File exists but service account lacks 'storage.objects.get' permission")
            print(f"  3. File path in console might be slightly different")
            print(f"\n  Action: Verify the exact path in Google Cloud Console")
            print(f"  Expected: gs://{bucket_name}/{target_path}")
    
    # Test 4: Check if bucket is accessible
    print(f"\n{'='*70}")
    print("Bucket Access Test")
    print(f"{'='*70}\n")
    
    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "id": f"log_{int(datetime.now().timestamp() * 1000)}",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "location": "debug_gcs_download.py:295",
            "message": "Testing bucket access",
            "data": {"bucket_name": bucket_name, "hypothesisId": "D"},
            "sessionId": "debug-session",
            "runId": "run1"
        }) + "\n")
    # #endregion
    
    try:
        bucket.reload()
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "id": f"log_{int(datetime.now().timestamp() * 1000)}",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "location": "debug_gcs_download.py:305",
                "message": "Bucket access successful",
                "data": {"hypothesisId": "D"},
                "sessionId": "debug-session",
                "runId": "run1"
            }) + "\n")
        # #endregion
        print(f"✓ Bucket '{bucket_name}' is accessible")
    except Exception as e:
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "id": f"log_{int(datetime.now().timestamp() * 1000)}",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "location": "debug_gcs_download.py:315",
                "message": "Bucket access failed",
                "data": {"error": str(e), "hypothesisId": "D"},
                "sessionId": "debug-session",
                "runId": "run1"
            }) + "\n")
        # #endregion
        print(f"✗ Cannot access bucket metadata: {e}")
        print(f"  Note: This doesn't prevent file downloads if you have 'storage.objects.get' permission")
    
    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "id": f"log_{int(datetime.now().timestamp() * 1000)}",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "location": "debug_gcs_download.py:225",
            "message": "Script completed",
            "data": {"hypothesisId": "A"},
            "sessionId": "debug-session",
            "runId": "run1"
        }) + "\n")
    # #endregion
    
    print(f"\n{'='*70}")
    print("Diagnostic complete!")
    print(f"{'='*70}\n")

except ImportError:
    print("✗ google-cloud-storage not installed")
    print("Install with: pip install google-cloud-storage")
except Exception as e:
    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "id": f"log_{int(datetime.now().timestamp() * 1000)}",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "location": "debug_gcs_download.py:245",
            "message": "Script error",
            "data": {"error": str(e), "error_type": type(e).__name__, "hypothesisId": "E"},
            "sessionId": "debug-session",
            "runId": "run1"
        }) + "\n")
    # #endregion
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

