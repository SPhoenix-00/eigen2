"""
Cloud storage synchronization utilities for checkpoints and logs.
Supports AWS S3, Google Cloud Storage, and Azure Blob Storage.
"""

import os
import json
import shutil
import threading
from pathlib import Path
from typing import Optional, Literal
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue

CloudProvider = Literal["s3", "gcs", "azure", "local"]


class CloudSync:
    """
    Handles checkpoint and log synchronization with cloud storage.
    """

    def __init__(
        self,
        provider: CloudProvider = "local",
        bucket_name: Optional[str] = None,
        project_name: str = "eigen2",
        credentials_path: Optional[str] = None,
        max_workers: int = 4
    ):
        """
        Initialize cloud sync.

        Args:
            provider: Cloud provider ('s3', 'gcs', 'azure', or 'local')
            bucket_name: Name of the cloud storage bucket
            project_name: Project identifier for organizing files
            credentials_path: Path to credentials file (for GCS)
            max_workers: Number of parallel upload threads
        """
        self.provider = provider
        self.bucket_name = bucket_name
        self.project_name = project_name
        self.credentials_path = credentials_path
        self.client = None

        # Background upload thread pool
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="CloudUpload")
        self.upload_futures = []
        self._lock = threading.Lock()

        if provider != "local":
            self._init_client()

    def _init_client(self):
        """Initialize the appropriate cloud storage client."""
        if self.provider == "s3":
            try:
                import boto3
                self.client = boto3.client('s3')
                print(f"✓ Connected to AWS S3 bucket: {self.bucket_name}")
            except ImportError:
                print("Warning: boto3 not installed. Install with: pip install boto3")
                self.provider = "local"
            except Exception as e:
                print(f"Warning: Could not connect to S3: {e}")
                print("Falling back to local storage only")
                self.provider = "local"

        elif self.provider == "gcs":
            try:
                from google.cloud import storage
                if self.credentials_path:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
                self.client = storage.Client()
                self.bucket = self.client.bucket(self.bucket_name)
                print(f"✓ Connected to Google Cloud Storage bucket: {self.bucket_name}")
            except ImportError:
                print("Warning: google-cloud-storage not installed. Install with: pip install google-cloud-storage")
                self.provider = "local"
            except Exception as e:
                print(f"Warning: Could not connect to GCS: {e}")
                print("Falling back to local storage only")
                self.provider = "local"

        elif self.provider == "azure":
            try:
                from azure.storage.blob import BlobServiceClient
                connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
                if not connection_string:
                    raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set")
                self.client = BlobServiceClient.from_connection_string(connection_string)
                self.container_client = self.client.get_container_client(self.bucket_name)
                print(f"✓ Connected to Azure Blob Storage container: {self.bucket_name}")
            except ImportError:
                print("Warning: azure-storage-blob not installed. Install with: pip install azure-storage-blob")
                self.provider = "local"
            except Exception as e:
                print(f"Warning: Could not connect to Azure: {e}")
                print("Falling back to local storage only")
                self.provider = "local"

    def _upload_file_sync(self, local_path: str, cloud_path: str):
        """
        Internal synchronous upload method (runs in background thread).

        Args:
            local_path: Path to local file
            cloud_path: Path in cloud storage
        """
        try:
            if self.provider == "s3":
                self.client.upload_file(local_path, self.bucket_name, cloud_path)

            elif self.provider == "gcs":
                blob = self.bucket.blob(cloud_path)
                blob.upload_from_filename(local_path)

            elif self.provider == "azure":
                blob_client = self.container_client.get_blob_client(cloud_path)
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)

            print(f"✓ Uploaded: {local_path} → {cloud_path}")

        except Exception as e:
            print(f"Warning: Failed to upload {local_path}: {e}")

    def upload_file(self, local_path: str, cloud_path: Optional[str] = None, background: bool = False):
        """
        Upload a file to cloud storage.

        Args:
            local_path: Path to local file
            cloud_path: Path in cloud storage (defaults to same as local)
            background: If True, upload in background thread (non-blocking)
        """
        if self.provider == "local":
            return

        if not os.path.exists(local_path):
            print(f"Warning: File not found: {local_path}")
            return

        if cloud_path is None:
            cloud_path = f"{self.project_name}/{local_path}"

        if background:
            # Submit to background thread pool
            future = self.executor.submit(self._upload_file_sync, local_path, cloud_path)
            with self._lock:
                self.upload_futures.append(future)
            print(f"⏳ Queued for upload: {local_path} → {cloud_path}")
        else:
            # Synchronous upload
            self._upload_file_sync(local_path, cloud_path)

    def file_exists_on_cloud(self, filename: str) -> bool:
        """
        Check if a file exists in cloud storage.

        Args:
            filename: Name of file (will be prefixed with project_name/checkpoints/)

        Returns:
            True if file exists, False otherwise
        """
        if self.provider == "local":
            return False

        cloud_path = f"{self.project_name}/checkpoints/{filename}"

        try:
            if self.provider == "s3":
                try:
                    self.client.head_object(Bucket=self.bucket_name, Key=cloud_path)
                    return True
                except:
                    return False

            elif self.provider == "gcs":
                blob = self.bucket.blob(cloud_path)
                return blob.exists()

            elif self.provider == "azure":
                blob_client = self.container_client.get_blob_client(cloud_path)
                return blob_client.exists()

        except Exception as e:
            print(f"Warning: Error checking if file exists: {e}")
            return False

        return False

    def download_file(self, cloud_path: str, local_path: str):
        """
        Download a file from cloud storage.

        Args:
            cloud_path: Path in cloud storage
            local_path: Path to save locally
        """
        if self.provider == "local":
            return False

        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            if self.provider == "s3":
                self.client.download_file(self.bucket_name, cloud_path, local_path)

            elif self.provider == "gcs":
                blob = self.bucket.blob(cloud_path)
                blob.download_to_filename(local_path)

            elif self.provider == "azure":
                blob_client = self.container_client.get_blob_client(cloud_path)
                with open(local_path, "wb") as f:
                    download_stream = blob_client.download_blob()
                    f.write(download_stream.readall())

            print(f"✓ Downloaded: {cloud_path} → {local_path}")
            return True

        except Exception as e:
            print(f"Warning: Failed to download {cloud_path}: {e}")
            return False

    def wait_for_uploads(self, timeout: Optional[float] = None):
        """
        Wait for all background uploads to complete.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
        """
        with self._lock:
            futures = list(self.upload_futures)

        if not futures:
            return

        print(f"\n⏳ Waiting for {len(futures)} background uploads to complete...")

        completed = 0
        failed = 0
        for future in futures:
            try:
                future.result(timeout=timeout)
                completed += 1
            except Exception as e:
                failed += 1
                print(f"Warning: Background upload failed: {e}")

        # Clear completed futures
        with self._lock:
            self.upload_futures = [f for f in self.upload_futures if not f.done()]

        print(f"✓ Background uploads complete: {completed} succeeded, {failed} failed")

    def get_upload_status(self):
        """
        Get status of background uploads.

        Returns:
            Tuple of (pending, completed, failed)
        """
        with self._lock:
            futures = list(self.upload_futures)

        pending = 0
        completed = 0
        failed = 0

        for future in futures:
            if future.done():
                try:
                    future.result(timeout=0)
                    completed += 1
                except:
                    failed += 1
            else:
                pending += 1

        return (pending, completed, failed)

    def upload_directory(self, local_dir: str, cloud_prefix: Optional[str] = None, background: bool = False):
        """
        Upload an entire directory to cloud storage.

        Args:
            local_dir: Local directory path
            cloud_prefix: Prefix for cloud storage paths
            background: If True, upload files in background threads (non-blocking)
        """
        if self.provider == "local":
            return

        if not os.path.exists(local_dir):
            print(f"Warning: Directory not found: {local_dir}")
            return

        if cloud_prefix is None:
            cloud_prefix = f"{self.project_name}/{local_dir}"

        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                cloud_path = f"{cloud_prefix}/{relative_path}".replace("\\", "/")
                self.upload_file(local_path, cloud_path, background=background)

    def download_directory(self, cloud_prefix: str, local_dir: str):
        """
        Download an entire directory from cloud storage.

        Args:
            cloud_prefix: Prefix in cloud storage
            local_dir: Local directory to save to
        """
        if self.provider == "local":
            return False

        try:
            if self.provider == "s3":
                paginator = self.client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=cloud_prefix)

                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            cloud_path = obj['Key']
                            relative_path = os.path.relpath(cloud_path, cloud_prefix)
                            local_path = os.path.join(local_dir, relative_path)
                            self.download_file(cloud_path, local_path)

            elif self.provider == "gcs":
                blobs = self.bucket.list_blobs(prefix=cloud_prefix)
                for blob in blobs:
                    relative_path = os.path.relpath(blob.name, cloud_prefix)
                    local_path = os.path.join(local_dir, relative_path)
                    self.download_file(blob.name, local_path)

            elif self.provider == "azure":
                blob_list = self.container_client.list_blobs(name_starts_with=cloud_prefix)
                for blob in blob_list:
                    relative_path = os.path.relpath(blob.name, cloud_prefix)
                    local_path = os.path.join(local_dir, relative_path)
                    self.download_file(blob.name, local_path)

            return True

        except Exception as e:
            print(f"Warning: Failed to download directory {cloud_prefix}: {e}")
            return False

    def sync_checkpoints(self, checkpoint_dir: str = "checkpoints", background: bool = False):
        """
        Sync checkpoint directory to cloud storage.

        Args:
            checkpoint_dir: Local checkpoint directory
            background: If True, upload in background (non-blocking)
        """
        print(f"\n{'='*60}")
        if background:
            print("Queueing checkpoints for background upload...")
        else:
            print("Syncing checkpoints to cloud storage...")
        print(f"{'='*60}")
        self.upload_directory(checkpoint_dir, f"{self.project_name}/{checkpoint_dir}", background=background)
        if not background:
            print(f"{'='*60}\n")

    def download_checkpoints(self, checkpoint_dir: str = "checkpoints"):
        """
        Download checkpoints from cloud storage.

        Args:
            checkpoint_dir: Local checkpoint directory
        """
        print(f"\n{'='*60}")
        print("Downloading checkpoints from cloud storage...")
        print(f"{'='*60}")
        success = self.download_directory(
            f"{self.project_name}/{checkpoint_dir}",
            checkpoint_dir
        )
        print(f"{'='*60}\n")
        return success

    def sync_logs(self, log_dir: str = "logs", background: bool = False):
        """
        Sync log directory to cloud storage.

        Args:
            log_dir: Local log directory
            background: If True, upload in background (non-blocking)
        """
        if self.provider != "local":
            self.upload_directory(log_dir, f"{self.project_name}/{log_dir}", background=background)

    def save_and_upload_buffer(self, replay_buffer, local_path: str, cloud_path: str):
        """
        Save replay buffer to disk and upload in background (non-blocking).

        This runs the entire save+upload operation in a background thread so
        training can continue immediately.

        Args:
            replay_buffer: ReplayBuffer instance to save
            local_path: Local path to save buffer
            cloud_path: Cloud storage path for upload
        """
        def _save_and_upload():
            """Internal function to run in background thread."""
            try:
                # Save buffer to disk (with compression)
                replay_buffer.save(local_path)

                # Upload to cloud
                if self.provider != "local":
                    self._upload_file_sync(local_path, cloud_path)

            except Exception as e:
                print(f"Warning: Failed to save and upload buffer: {e}")

        # Submit to background thread pool
        future = self.executor.submit(_save_and_upload)
        with self._lock:
            self.upload_futures.append(future)

        buffer_size_mb = len(replay_buffer) * 3.74
        print(f"⏳ Queued buffer save+upload: {len(replay_buffer)} transitions (~{buffer_size_mb:.0f} MB)")

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """
        Shutdown the upload thread pool.

        Args:
            wait: If True, wait for pending uploads to complete
            timeout: Maximum time to wait in seconds
        """
        if wait:
            self.wait_for_uploads(timeout=timeout)
        self.executor.shutdown(wait=wait)


def get_cloud_sync_from_env() -> CloudSync:
    """
    Create CloudSync instance from environment variables.

    Environment variables:
        CLOUD_PROVIDER: 's3', 'gcs', 'azure', or 'local' (default: 'local')
        CLOUD_BUCKET: Bucket/container name
        CLOUD_PROJECT: Project name (default: 'eigen2')
        GOOGLE_APPLICATION_CREDENTIALS: Path to GCS credentials JSON (for GCS only)
        GCS_CREDENTIALS: Alternative path to GCS credentials (deprecated, use GOOGLE_APPLICATION_CREDENTIALS)

    Returns:
        Configured CloudSync instance
    """
    provider = os.environ.get("CLOUD_PROVIDER", "local").lower()
    bucket_name = os.environ.get("CLOUD_BUCKET")
    project_name = os.environ.get("CLOUD_PROJECT", "eigen2")
    # Check both GOOGLE_APPLICATION_CREDENTIALS (standard) and GCS_CREDENTIALS (legacy)
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GCS_CREDENTIALS")

    if provider != "local" and not bucket_name:
        print("Warning: CLOUD_BUCKET not set. Using local storage only.")
        provider = "local"

    return CloudSync(
        provider=provider,
        bucket_name=bucket_name,
        project_name=project_name,
        credentials_path=credentials_path
    )
