"""
Committee Manager: handles roster storage, cloud synchronization, and mirror checks.
"""

import os
import json
import numpy as np
import tempfile
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from committee.utils import (
    convert_numpy_types, load_global50_candidates, get_agent_filepath,
    GLOBAL50_BASE_DIR, ROSTER_FILENAME, CORRELATION_FILENAME,
)
from utils.cloud_sync import get_cloud_sync_from_env


class CommitteeManager:
    """
    Manages committee selection, storage, and cloud synchronization.
    Committee files live in global50/cw{N}/committee/ alongside agents/.
    """

    def __init__(self, context_window_days: int):
        """
        Initialize CommitteeManager.

        Args:
            context_window_days: Context window for this league (e.g., 151)
        """
        self.context_window_days = context_window_days
        self.context_window_id = f"cw{context_window_days}"

        # Cloud sync
        self.cloud_sync = get_cloud_sync_from_env()

        # Local paths
        self.local_base = GLOBAL50_BASE_DIR / self.context_window_id
        self.local_committee_dir = self.local_base / "committee"
        self.local_roster_path = self.local_committee_dir / ROSTER_FILENAME
        self.local_correlation_path = self.local_committee_dir / CORRELATION_FILENAME

        # Cloud paths
        self.cloud_base = f"{self.cloud_sync.project_name}/global50/{self.context_window_id}"
        self.cloud_committee_base = f"{self.cloud_base}/committee"
        self.cloud_roster_path = f"{self.cloud_committee_base}/{ROSTER_FILENAME}"
        self.cloud_correlation_path = f"{self.cloud_committee_base}/{CORRELATION_FILENAME}"

        # Ensure local directories exist
        self.local_committee_dir.mkdir(parents=True, exist_ok=True)

    def save_roster(self, roster_data: dict, correlation_matrix: np.ndarray = None) -> bool:
        """
        Save committee roster locally and sync to cloud.

        Args:
            roster_data: Committee roster dictionary
            correlation_matrix: Optional correlation matrix for heatmap

        Returns:
            True if save and sync succeeded
        """
        # Convert numpy types to native Python types for JSON serialization
        roster_data_clean = convert_numpy_types(roster_data)

        # Save roster JSON locally
        with open(self.local_roster_path, 'w') as f:
            json.dump(roster_data_clean, f, indent=2)
        print(f"✓ Roster saved locally: {self.local_roster_path}")

        # Save correlation heatmap if provided
        if correlation_matrix is not None:
            self._save_correlation_heatmap(roster_data, correlation_matrix)

        # Sync to cloud
        return self._sync_to_cloud()

    def _save_correlation_heatmap(self, roster_data: dict, correlation_matrix: np.ndarray):
        """Save correlation heatmap visualization."""
        plt.figure(figsize=(10, 8))
        labels = [f"{m['run_name']}_{m['agent_id']}"[:15] for m in roster_data['members']]

        avg_corr = roster_data.get('correlation', {}).get('average', 0)
        max_corr = roster_data.get('correlation', {}).get('max_pair', 0)

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                    vmin=-1, vmax=1, xticklabels=labels, yticklabels=labels)
        plt.title(f"Committee Coefficient Correlation\n"
                  f"(Avg: {avg_corr:.3f}, Max: {max_corr:.3f})")
        plt.tight_layout()
        plt.savefig(self.local_correlation_path, dpi=150)
        plt.close()
        print(f"✓ Correlation heatmap saved: {self.local_correlation_path}")

    def _sync_to_cloud(self) -> bool:
        """Sync local committee files to cloud storage."""
        if self.cloud_sync.provider == "local":
            print("  ⚠ Cloud sync disabled (local mode)")
            return True

        print(f"\nSyncing committee to cloud...")
        success = True

        # Upload roster JSON
        if self.local_roster_path.exists():
            if self.cloud_sync.upload_file_verified(
                str(self.local_roster_path), self.cloud_roster_path
            ):
                print(f"  ✓ Uploaded: {ROSTER_FILENAME}")
            else:
                print(f"  ✗ Failed to upload: {ROSTER_FILENAME}")
                success = False

        # Upload correlation heatmap
        if self.local_correlation_path.exists():
            if self.cloud_sync.upload_file_verified(
                str(self.local_correlation_path), self.cloud_correlation_path
            ):
                print(f"  ✓ Uploaded: {CORRELATION_FILENAME}")
            else:
                print(f"  ✗ Failed to upload: {CORRELATION_FILENAME}")
                success = False

        if success:
            print(f"  ✓ Cloud mirror: gs://{self.cloud_sync.bucket_name}/{self.cloud_committee_base}/")

        return success

    def load_roster(self) -> dict:
        """Load committee roster from local storage."""
        if not self.local_roster_path.exists():
            return None

        with open(self.local_roster_path, 'r') as f:
            return json.load(f)

    def update_maverick_flags(self) -> bool:
        """
        Update is_maverick flags in existing committee roster by syncing from Global50.

        Returns:
            True if update succeeded, False otherwise
        """
        print("\n" + "="*60)
        print(f"UPDATING MAVERICK FLAGS IN COMMITTEE ROSTER ({self.context_window_id})")
        print("="*60)

        # Load current roster
        roster = self.load_roster()
        if not roster:
            print("❌ No committee roster found. Run --draft first.")
            return False

        # Load Global50 entries
        entries = load_global50_candidates(self.context_window_days)
        if not entries:
            print("❌ No agents in Global50")
            return False

        # Create lookup dict: (run_name, agent_id) -> is_maverick
        global50_maverick_map = {}
        for entry in entries:
            key = (entry['run_name'], entry['agent_id'])
            global50_maverick_map[key] = entry.get('is_maverick', False)

        # Update roster members
        updated_count = 0
        members = roster.get('members', [])

        print(f"\nChecking {len(members)} committee members...")
        for member in members:
            run_name = member.get('run_name')
            agent_id = member.get('agent_id')
            key = (run_name, agent_id)

            if key in global50_maverick_map:
                new_maverick_flag = global50_maverick_map[key]
                old_maverick_flag = member.get('is_maverick', False)

                if new_maverick_flag != old_maverick_flag:
                    member['is_maverick'] = new_maverick_flag
                    updated_count += 1
                    action = "SET" if new_maverick_flag else "UNSET"
                    print(f"  {action}: {run_name}_{agent_id} -> is_maverick={new_maverick_flag}")
            else:
                print(f"  ⚠ {run_name}_{agent_id}: Not found in Global50 (may have been removed)")

        if updated_count == 0:
            print(f"\n✓ No updates needed - all maverick flags are already in sync")
            return True

        # Update timestamp
        roster['last_updated'] = datetime.now().isoformat() + 'Z'
        roster['maverick_flags_updated_at'] = datetime.now().isoformat() + 'Z'

        # Save updated roster
        print(f"\nSaving updated roster...")
        if self.save_roster(roster):
            print(f"✓ Updated {updated_count} member(s) and synced to cloud")

            # Print summary
            maverick_count = sum(1 for m in members if m.get('is_maverick', False))
            print(f"\n  Committee Summary:")
            print(f"    Total members: {len(members)}")
            print(f"    Maverick members: {maverick_count}")
            print(f"    Non-maverick members: {len(members) - maverick_count}")

            if maverick_count == 0:
                print(f"\n  ⚠ WARNING: No maverick agents in committee!")
                print(f"    --multi mode requires at least one maverick agent.")
                print(f"    Use fix_global50.py --set-maverick to mark agents as mavericks in Global50,")
                print(f"    then run --draft to create a new committee with mavericks.")

            return True
        else:
            print(f"✗ Failed to save updated roster")
            return False

    def check_mirror_status(self) -> bool:
        """
        Check synchronization status between local and cloud.
        Downloads missing files from cloud if needed.

        Returns:
            True if in sync, False if mismatch or error
        """
        print("\n" + "="*60)
        print(f"COMMITTEE MIRROR CHECK ({self.context_window_id})")
        print("="*60)

        if self.cloud_sync.provider == "local":
            print("  ⚠ Cloud sync disabled (local mode)")
            print("  Checking local files only...")
            return self._check_local_status()

        print(f"  Cloud base: gs://{self.cloud_sync.bucket_name}/{self.cloud_committee_base}/")
        print(f"  Local base: {self.local_committee_dir}/")

        # Check cloud roster exists
        cloud_roster_exists = self._cloud_file_exists(self.cloud_roster_path)
        local_roster_exists = self.local_roster_path.exists()

        print(f"\n  Roster JSON:")
        print(f"    Cloud: {'EXISTS' if cloud_roster_exists else 'MISSING'}")
        print(f"    Local: {'EXISTS' if local_roster_exists else 'MISSING'}")

        # Download from cloud if local missing
        if cloud_roster_exists and not local_roster_exists:
            print(f"  → Downloading roster from cloud...")
            if self.cloud_sync.download_file(self.cloud_roster_path, str(self.local_roster_path)):
                print(f"    ✓ Downloaded {ROSTER_FILENAME}")
                local_roster_exists = True
            else:
                print(f"    ✗ Failed to download {ROSTER_FILENAME}")

        # Check correlation heatmap
        cloud_corr_exists = self._cloud_file_exists(self.cloud_correlation_path)
        local_corr_exists = self.local_correlation_path.exists()

        print(f"\n  Correlation Heatmap:")
        print(f"    Cloud: {'EXISTS' if cloud_corr_exists else 'MISSING'}")
        print(f"    Local: {'EXISTS' if local_corr_exists else 'MISSING'}")

        if cloud_corr_exists and not local_corr_exists:
            print(f"  → Downloading correlation heatmap from cloud...")
            if self.cloud_sync.download_file(self.cloud_correlation_path, str(self.local_correlation_path)):
                print(f"    ✓ Downloaded {CORRELATION_FILENAME}")
                local_corr_exists = True
            else:
                print(f"    ✗ Failed to download {CORRELATION_FILENAME}")

        # Compare local vs cloud if both exist
        if cloud_roster_exists and local_roster_exists:
            match = self._compare_rosters()
            if match:
                print(f"\n  ✓ Local and cloud rosters MATCH")
            else:
                print(f"\n  ⚠ Local and cloud rosters DIFFER")
                print(f"    Use --draft to regenerate committee")
                return False

        # Check and mirror agent files
        agent_sync_success = True
        roster = None
        if local_roster_exists:
            roster = self.load_roster()
            members = roster.get('members', [])

            if members:
                print(f"\n  Agent Files ({len(members)} members):")
                agents_dir = self.local_base / "agents"
                agents_dir.mkdir(parents=True, exist_ok=True)

                cloud_agents_base = f"{self.cloud_base}/agents"

                downloaded_count = 0
                uploaded_count = 0
                missing_local = []
                missing_cloud = []

                for member in members:
                    run_name = member.get('run_name')
                    agent_id = member.get('agent_id')

                    if not run_name or not agent_id:
                        print(f"    ⚠ Skipping member with missing run_name or agent_id")
                        continue

                    # Get file paths
                    local_agent_path = get_agent_filepath(member, self.context_window_days)
                    filename = local_agent_path.name
                    cloud_agent_path = f"{cloud_agents_base}/{filename}"

                    local_exists = local_agent_path.exists()
                    cloud_exists = self._cloud_file_exists(cloud_agent_path)

                    # Download from cloud if local missing
                    if cloud_exists and not local_exists:
                        print(f"    → Downloading: {filename}")
                        local_agent_path.parent.mkdir(parents=True, exist_ok=True)
                        if self.cloud_sync.download_file(cloud_agent_path, str(local_agent_path)):
                            print(f"      ✓ Downloaded {filename}")
                            downloaded_count += 1
                        else:
                            print(f"      ✗ Failed to download {filename}")
                            agent_sync_success = False
                            missing_local.append(filename)
                    elif not cloud_exists and not local_exists:
                        print(f"    ✗ Missing both local and cloud: {filename}")
                        missing_local.append(filename)
                        agent_sync_success = False
                    elif local_exists and not cloud_exists:
                        # Upload to cloud if cloud missing (optional but helpful)
                        print(f"    → Uploading: {filename}")
                        if self.cloud_sync.upload_file_verified(str(local_agent_path), cloud_agent_path):
                            print(f"      ✓ Uploaded {filename}")
                            uploaded_count += 1
                        else:
                            print(f"      ✗ Failed to upload {filename}")
                            missing_cloud.append(filename)
                    # else: both exist, no action needed

                # Summary
                if downloaded_count > 0 or uploaded_count > 0:
                    print(f"\n  Agent Sync Summary:")
                    if downloaded_count > 0:
                        print(f"    ✓ Downloaded: {downloaded_count} agent(s)")
                    if uploaded_count > 0:
                        print(f"    ✓ Uploaded: {uploaded_count} agent(s)")

                if missing_local or missing_cloud:
                    if missing_local:
                        print(f"    ⚠ Missing locally: {len(missing_local)} agent(s)")
                    if missing_cloud:
                        print(f"    ⚠ Missing in cloud: {len(missing_cloud)} agent(s)")

        # Summary
        if roster:
            print(f"\n  Committee Status:")
            print(f"    Size: {roster.get('committee_size', 'N/A')}")
            print(f"    Objective: {roster.get('objective_value', 'N/A'):.2f}")
            print(f"    Avg Correlation: {roster.get('correlation', {}).get('average', 'N/A'):.3f}")
            print(f"    Generated: {roster.get('generated_at', 'N/A')}")
            return agent_sync_success
        else:
            print(f"\n  ⚠ No committee roster found")
            print(f"    Run --draft to create committee")
            return False

    def _check_local_status(self) -> bool:
        """Check local committee status when cloud is disabled."""
        if self.local_roster_path.exists():
            roster = self.load_roster()

            # Check agent files
            members = roster.get('members', [])
            if members:
                print(f"\n  Agent Files ({len(members)} members):")
                missing_count = 0

                for member in members:
                    run_name = member.get('run_name')
                    agent_id = member.get('agent_id')

                    if not run_name or not agent_id:
                        continue

                    local_agent_path = get_agent_filepath(member, self.context_window_days)
                    if not local_agent_path.exists():
                        print(f"    ✗ Missing: {local_agent_path.name}")
                        missing_count += 1

                if missing_count == 0:
                    print(f"    ✓ All {len(members)} agent files present")
                else:
                    print(f"    ⚠ {missing_count} agent file(s) missing")

            print(f"\n  Committee Status:")
            print(f"    Size: {roster.get('committee_size', 'N/A')}")
            print(f"    Objective: {roster.get('objective_value', 'N/A'):.2f}")
            print(f"    Avg Correlation: {roster.get('correlation', {}).get('average', 'N/A'):.3f}")
            print(f"    Generated: {roster.get('generated_at', 'N/A')}")
            return missing_count == 0 if members else True
        else:
            print(f"\n  ⚠ No committee roster found locally")
            print(f"    Run --draft to create committee")
            return False

    def _cloud_file_exists(self, cloud_path: str) -> bool:
        """Check if a file exists in cloud storage."""
        try:
            return self.cloud_sync.file_exists(cloud_path)
        except Exception:
            return False

    def _compare_rosters(self) -> bool:
        """Compare local and cloud rosters for equality."""
        temp_path = None
        try:
            # Download cloud roster to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                temp_path = tmp.name

            if not self.cloud_sync.download_file(self.cloud_roster_path, temp_path, silent=True):
                return False

            with open(temp_path, 'r') as f:
                cloud_roster = json.load(f)

            local_roster = self.load_roster()

            # Compare key fields
            if local_roster.get('committee_size') != cloud_roster.get('committee_size'):
                return False

            local_members = set(m['filename'] for m in local_roster.get('members', []))
            cloud_members = set(m['filename'] for m in cloud_roster.get('members', []))

            return local_members == cloud_members

        except Exception as e:
            print(f"  ⚠ Error comparing rosters: {e}")
            return False
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

