#!/usr/bin/env python3
"""
Quick test script to verify GCS connectivity before training.
Run this to ensure credentials and environment are set up correctly.
"""

import os
import sys
from pathlib import Path

def test_environment_variables():
    """Test if required environment variables are set."""
    print("=" * 60)
    print("1. Checking Environment Variables")
    print("=" * 60)

    required_vars = {
        'CLOUD_PROVIDER': 'gcs',
        'CLOUD_BUCKET': 'eigen2-checkpoints-ase0',
        'GOOGLE_APPLICATION_CREDENTIALS': '/workspace/gcs-credentials.json'
    }

    all_ok = True
    for var, expected in required_vars.items():
        actual = os.environ.get(var)
        if actual:
            status = "✓" if actual == expected else "⚠️"
            print(f"{status} {var} = {actual}")
            if actual != expected:
                print(f"   Expected: {expected}")
                all_ok = False
        else:
            print(f"✗ {var} = NOT SET")
            print(f"   Expected: {expected}")
            all_ok = False

    print()
    return all_ok

def test_credentials_file():
    """Test if credentials file exists and is valid JSON."""
    print("=" * 60)
    print("2. Checking Credentials File")
    print("=" * 60)

    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not creds_path:
        print("✗ GOOGLE_APPLICATION_CREDENTIALS not set")
        print()
        return False

    if not os.path.exists(creds_path):
        print(f"✗ File not found: {creds_path}")
        print()
        return False

    print(f"✓ File exists: {creds_path}")

    # Try to read as JSON
    try:
        import json
        with open(creds_path, 'r') as f:
            data = json.load(f)
        print(f"✓ Valid JSON")
        print(f"✓ Project ID: {data.get('project_id', 'N/A')}")
        print(f"✓ Client Email: {data.get('client_email', 'N/A')}")
    except Exception as e:
        print(f"✗ Error reading JSON: {e}")
        print()
        return False

    print()
    return True

def test_google_cloud_library():
    """Test if google-cloud-storage is installed."""
    print("=" * 60)
    print("3. Checking Google Cloud Storage Library")
    print("=" * 60)

    try:
        from google.cloud import storage
        print("✓ google-cloud-storage installed")
        print(f"✓ Version: {storage.__version__}")
    except ImportError as e:
        print("✗ google-cloud-storage not installed")
        print("  Run: pip install google-cloud-storage")
        print()
        return False

    print()
    return True

def test_gcs_connection():
    """Test actual connection to GCS."""
    print("=" * 60)
    print("4. Testing GCS Connection")
    print("=" * 60)

    try:
        from google.cloud import storage

        bucket_name = os.environ.get('CLOUD_BUCKET')
        print(f"Connecting to bucket: {bucket_name}")

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Test if bucket exists (note: requires storage.buckets.get permission)
        try:
            if bucket.exists():
                print(f"✓ Connected to bucket: {bucket_name}")
        except Exception as perm_error:
            # Permission denied on bucket.exists() is OK - we don't need it for training
            if "403" in str(perm_error) or "storage.buckets.get" in str(perm_error):
                print(f"⚠️  Cannot verify bucket exists (missing storage.buckets.get permission)")
                print(f"   This is OK - not needed for checkpoint uploads")
            else:
                raise

        # Try to list some blobs (limit to 5)
        print(f"✓ Listing files in bucket...")
        try:
            blobs = list(bucket.list_blobs(max_results=5))
            if blobs:
                print(f"✓ Found {len(blobs)} files (showing first 5):")
                for blob in blobs[:5]:
                    print(f"  - {blob.name}")
            else:
                print("  (Bucket is empty or no list permission)")
        except Exception as list_error:
            print(f"⚠️  Cannot list files: {list_error}")
            print("   This is OK if write/read test passes")

    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print()
        return False

    print()
    return True

def test_write_and_read():
    """Test writing and reading a file to/from GCS."""
    print("=" * 60)
    print("5. Testing Write/Read Operations")
    print("=" * 60)

    try:
        from google.cloud import storage

        bucket_name = os.environ.get('CLOUD_BUCKET')
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Write test file
        test_blob_name = "eigen2/test_connection.txt"
        test_content = "GCS connection test - OK!"

        print(f"Writing test file: {test_blob_name}")
        blob = bucket.blob(test_blob_name)
        blob.upload_from_string(test_content)
        print("✓ Write successful")

        # Read test file
        print(f"Reading test file: {test_blob_name}")
        downloaded_content = blob.download_as_string().decode('utf-8')

        if downloaded_content == test_content:
            print("✓ Read successful")
            print(f"✓ Content matches: '{downloaded_content}'")
        else:
            print("✗ Content mismatch")
            print(f"  Expected: '{test_content}'")
            print(f"  Got: '{downloaded_content}'")
            print()
            return False

        # Clean up test file
        print(f"Deleting test file...")
        blob.delete()
        print("✓ Cleanup successful")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        print()
        return False

    print()
    return True

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GCS Connection Test for Eigen2")
    print("=" * 60 + "\n")

    tests = [
        ("Environment Variables", test_environment_variables),
        ("Credentials File", test_credentials_file),
        ("Google Cloud Library", test_google_cloud_library),
        ("GCS Connection", test_gcs_connection),
        ("Write/Read Operations", test_write_and_read),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Unexpected error in {name}: {e}\n")
            results.append((name, False))

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
        if not result:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! GCS is ready for training.")
        print("\nYou can now run: python main.py")
        return 0
    else:
        print("\n✗ Some tests failed. Fix the issues above before training.")
        print("\nCommon fixes:")
        print("  1. Set environment variables:")
        print("     export CLOUD_PROVIDER=gcs")
        print("     export CLOUD_BUCKET=eigen2-checkpoints-ase0")
        print("     export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcs-credentials.json")
        print("  2. Install library:")
        print("     pip install google-cloud-storage")
        print("  3. Check credentials file exists and is valid JSON")
        return 1

if __name__ == "__main__":
    sys.exit(main())
