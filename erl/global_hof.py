"""
Global Hall of Fame - Cross-Run Top 50 Tracking
Maintains a persistent, global leaderboard of the best agents across all training runs.
"""

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from models.ddpg_agent import DDPGAgent


@dataclass
class LeagueRules:
    """
    Configuration rules that define the competitive league.
    All runs must match these rules to participate in the Global 50.
    """
    context_window_days: int
    # Add other league-defining parameters here as needed
    # e.g., episode_length, min_holding_period, etc.

    def matches(self, other: 'LeagueRules') -> bool:
        """Check if this league config matches another."""
        return self.context_window_days == other.context_window_days

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> 'LeagueRules':
        """Create from dictionary."""
        return LeagueRules(**data)


@dataclass
class GlobalHoFEntry:
    """
    A single entry in the Global Hall of Fame.
    """
    agent_id: int
    run_name: str
    gauntlet_score: float
    generation: int
    roi: float = 0.0
    expectancy: float = 0.0
    quality_ratio: float = 0.0
    win_ratio: float = 0.0
    total_trades: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> 'GlobalHoFEntry':
        """Create from dictionary with backward compatibility."""
        # Handle migration from old format (quality_count) to new format (quality_ratio, win_ratio)
        if 'quality_count' in data and 'quality_ratio' not in data:
            # Old format detected - convert to new format
            quality_count = data.pop('quality_count')
            total_trades = data.get('total_trades', 0)

            # Calculate quality_ratio from quality_count
            data['quality_ratio'] = float(quality_count / total_trades) if total_trades > 0 else 0.0

            # win_ratio wasn't stored before, default to 0.0
            data['win_ratio'] = 0.0

        return GlobalHoFEntry(**data)

    def get_filename(self) -> str:
        """Generate unique filename for this agent's weights."""
        # Format: [RunName]_[AgentID].pth
        # Note: Do NOT include score - it changes during re-evaluation!
        return f"{self.run_name}_{self.agent_id}.pth"


class GlobalHallOfFame:
    """
    Global Hall of Fame - Tracks the top 50 agents ever produced across all training runs.

    This system operates in real-time:
    - Every run is aware of the global entry threshold (score of rank #50)
    - Agents are promoted immediately upon passing the Gauntlet if they qualify
    - Handles concurrency via atomic updates (re-download before write)
    """

    CAPACITY = 50
    LOCAL_BASE_DIR = Path("global50")

    def __init__(self, cloud_sync, run_name: str, league_rules: LeagueRules,
                 checkpoint_dir: Path, disable_global50: bool = False):
        """
        Initialize Global Hall of Fame.

        Args:
            cloud_sync: CloudSync instance for GCS operations
            run_name: Current training run name (e.g., "azure-thunder-123")
            league_rules: Configuration rules that define this league
            checkpoint_dir: Local checkpoint directory for this run
            disable_global50: If True, disable Global 50 for this run (read-only mode)
        """
        self.cloud_sync = cloud_sync
        self.run_name = run_name
        self.league_rules = league_rules
        self.checkpoint_dir = checkpoint_dir
        self.enabled = not disable_global50 and (cloud_sync.provider != "local")

        # Local cache
        self.entries: List[GlobalHoFEntry] = []
        self.entry_threshold: float = float('-inf')
        self.league_compatible: bool = False

        # Thread safety for atomic updates
        self._lock = threading.Lock()

        # Context window identifier for path (e.g., "cw151" for 151 days)
        self.context_window_id = f"cw{league_rules.context_window_days}"

        # Paths (now include context window subdirectory)
        self.local_dir = self.LOCAL_BASE_DIR / self.context_window_id
        self.local_agents_dir = self.local_dir / "agents"
        self.local_archive_dir = self.local_dir / "archive"
        self.local_json_path = self.local_dir / "global50.json"

        # Cloud paths (now include context window subdirectory)
        self.cloud_base = f"{cloud_sync.project_name}/global50/{self.context_window_id}"
        self.cloud_json_path = f"{self.cloud_base}/global50.json"

        # Fallback leagues for diversity injection (populated during initialization)
        self.fallback_leagues: List[Dict] = []

        if self.enabled:
            self._initialize()

    def _initialize(self):
        """
        Phase A: Startup
        - Create local directories
        - Download global50.json from cloud for our context window
        - If no match, create new league for this context window
        - Discover other context windows for fallback diversity injection
        """
        # Create local directories
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.local_agents_dir.mkdir(parents=True, exist_ok=True)
        self.local_archive_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print("Global Hall of Fame - Initialization")
        print(f"{'='*60}")
        print(f"Context Window: {self.league_rules.context_window_days} days")

        # Download global50.json for our context window
        success = self._download_global_ledger()

        if success:
            # Load and validate
            self._load_local_ledger()
            self._validate_league_rules()
            self._update_entry_threshold()
            print(f"✓ Connected to existing Global 50 (context window: {self.league_rules.context_window_days} days)")
        else:
            # No existing global50.json for this context window - create new league
            print(f"⚠ No existing Global 50 for {self.league_rules.context_window_days} days context window.")
            print(f"  This will be the founding run for this context window.")
            self.league_compatible = True
            self.entry_threshold = float('-inf')
            # Initialize empty ledger
            self._save_local_ledger()
            self._upload_global_ledger()

        if self.league_compatible:
            print(f"✓ League Validation: PASSED")
            print(f"✓ Entry Threshold: {self.entry_threshold:.2f} (Rank #50)")
            print(f"✓ Current Global 50 Size: {len(self.entries)}")
        else:
            print(f"✗ League Validation: FAILED")
            print(f"✗ Global 50 DISABLED for this run to prevent pollution")
            self.enabled = False

        # Discover other context windows for fallback diversity injection
        if self.enabled:
            self._discover_fallback_leagues()

        print(f"{'='*60}\n")

    def _download_global_ledger(self) -> bool:
        """Download global50.json from cloud storage."""
        try:
            return self.cloud_sync.download_file(
                self.cloud_json_path,
                str(self.local_json_path)
            )
        except Exception as e:
            print(f"⚠ Could not download global50.json: {e}")
            return False

    def _upload_global_ledger(self) -> bool:
        """Upload global50.json to cloud storage."""
        try:
            self.cloud_sync.upload_file(
                str(self.local_json_path),
                self.cloud_json_path,
                background=False  # Synchronous for ledger updates
            )
            return True
        except Exception as e:
            print(f"⚠ Failed to upload global50.json: {e}")
            return False

    def _load_local_ledger(self):
        """Load global50.json from local cache."""
        if not self.local_json_path.exists():
            return

        with open(self.local_json_path, 'r') as f:
            data = json.load(f)

        # Load league rules with backward compatibility
        # If no league_rules specified, assume old default context window of 504 days
        league_rules_data = data.get('league_rules', {})
        if 'context_window_days' not in league_rules_data:
            league_rules_data['context_window_days'] = 504  # Old default
        stored_rules = LeagueRules.from_dict(league_rules_data)

        # Load entries
        self.entries = [GlobalHoFEntry.from_dict(e) for e in data.get('entries', [])]

        # Store for validation
        self._stored_league_rules = stored_rules

    def _save_local_ledger(self):
        """Save global50.json to local cache."""
        data = {
            'league_rules': self.league_rules.to_dict(),
            'entries': [e.to_dict() for e in self.entries],
            'capacity': self.CAPACITY,
            'version': '1.0'
        }

        with open(self.local_json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _validate_league_rules(self):
        """Validate that current run's config matches the Global League rules."""
        if not hasattr(self, '_stored_league_rules'):
            # No stored rules (first run)
            self.league_compatible = True
            return

        if self.league_rules.matches(self._stored_league_rules):
            self.league_compatible = True
        else:
            self.league_compatible = False
            print(f"\n⚠ LEAGUE MISMATCH DETECTED:")
            print(f"  Current: {self.league_rules.to_dict()}")
            print(f"  Global:  {self._stored_league_rules.to_dict()}")

    def _update_entry_threshold(self):
        """Update the local entry threshold (score of rank #50)."""
        if len(self.entries) < self.CAPACITY:
            self.entry_threshold = float('-inf')
        else:
            # Get the 50th ranked agent's score (worst in top 50)
            sorted_entries = sorted(self.entries, key=lambda e: e.gauntlet_score, reverse=True)
            self.entry_threshold = sorted_entries[self.CAPACITY - 1].gauntlet_score

    def _discover_fallback_leagues(self):
        """
        Discover other context window leagues available in cloud storage.
        These can be used for diversity injection if our league is empty or struggling.
        """
        print(f"Discovering other context window leagues for diversity injection...")

        # First, check for old global50 structure (without context window subdirectory)
        # This would be the legacy 504-day context window league
        if self.league_rules.context_window_days != 504:
            old_cloud_json_path = f"{self.cloud_sync.project_name}/global50/global50.json"
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=True) as tmp:
                try:
                    success = self.cloud_sync.download_file(old_cloud_json_path, tmp.name)
                    if success:
                        # Load to get entry count
                        with open(tmp.name, 'r') as f:
                            data = json.load(f)
                        entry_count = len(data.get('entries', []))

                        self.fallback_leagues.append({
                            'context_window_days': 504,
                            'context_window_id': 'legacy',  # Special ID for old structure
                            'entry_count': entry_count,
                            'cloud_base': f"{self.cloud_sync.project_name}/global50",  # No subdirectory
                            'is_legacy': True
                        })
                        print(f"  ✓ Found legacy fallback: 504 days (legacy structure, {entry_count} agents)")
                except Exception:
                    pass  # Doesn't exist, skip

        # Dynamically discover available context window leagues from GCS
        available_windows = []

        if self.cloud_sync.provider == "gcs":
            try:
                # List all directories under eigen2/global50/
                prefix = f"{self.cloud_sync.project_name}/global50/"
                blobs = self.cloud_sync.bucket.list_blobs(prefix=prefix, delimiter='/')

                # Extract context window IDs from directory names
                for page in blobs.pages:
                    for prefix_path in page.prefixes:
                        # Extract directory name (e.g., "cw504" from "eigen2/global50/cw504/")
                        dir_name = prefix_path.rstrip('/').split('/')[-1]
                        if dir_name.startswith('cw'):
                            try:
                                window_days = int(dir_name[2:])  # Extract number from "cw504"
                                if window_days != self.league_rules.context_window_days:
                                    available_windows.append(window_days)
                            except ValueError:
                                pass  # Skip if not a valid number

                print(f"  Found {len(available_windows)} context window leagues in bucket: {sorted(available_windows)}")
            except Exception as e:
                print(f"  Warning: Could not list bucket directories: {e}")
                print(f"  Falling back to default common windows")
                # Fallback to common windows if bucket listing fails
                available_windows = [504, 252, 377, 125, 100, 200, 300]
                available_windows = [w for w in available_windows if w != self.league_rules.context_window_days]
        else:
            # For non-GCS providers, use common windows as fallback
            available_windows = [504, 252, 377, 125, 100, 200, 300]
            available_windows = [w for w in available_windows if w != self.league_rules.context_window_days]

        # Try to download JSON for each discovered context window
        import tempfile
        for window_days in available_windows:
            fallback_id = f"cw{window_days}"
            cloud_json_path = f"{self.cloud_sync.project_name}/global50/{fallback_id}/global50.json"

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=True) as tmp:
                try:
                    success = self.cloud_sync.download_file(cloud_json_path, tmp.name)
                    if success:
                        # Load to get entry count
                        with open(tmp.name, 'r') as f:
                            data = json.load(f)
                        entry_count = len(data.get('entries', []))

                        self.fallback_leagues.append({
                            'context_window_days': window_days,
                            'context_window_id': fallback_id,
                            'entry_count': entry_count,
                            'cloud_base': f"{self.cloud_sync.project_name}/global50/{fallback_id}",
                            'is_legacy': False
                        })
                        print(f"  ✓ Found fallback: {window_days} days ({entry_count} agents)")
                except Exception:
                    pass  # Doesn't exist, skip

        if len(self.fallback_leagues) == 0:
            print(f"  No fallback leagues found (only {self.league_rules.context_window_days} days league exists)")
        else:
            print(f"  Total fallback leagues: {len(self.fallback_leagues)}")

    def should_promote(self, gauntlet_score: float) -> bool:
        """
        Check if an agent with given gauntlet score qualifies for Global 50.

        Args:
            gauntlet_score: Agent's certified Gauntlet score

        Returns:
            True if agent should be promoted, False otherwise
        """
        if not self.enabled or not self.league_compatible:
            return False

        return gauntlet_score > self.entry_threshold

    def check_and_promote(self, agent: DDPGAgent, gauntlet_score: float, generation: int,
                          roi: float = 0.0, expectancy: float = 0.0,
                          quality_ratio: float = 0.0, win_ratio: float = 0.0,
                          total_trades: int = 0) -> bool:
        """
        Phase C: The Promotion Routine (Atomic Update)

        Promotes an agent to the Global 50 if they qualify.
        Handles concurrency by re-downloading the ledger before writing.

        Args:
            agent: The DDPGAgent instance to promote
            gauntlet_score: Certified Gauntlet score (DO NOT re-test)
            generation: Generation when agent passed Gauntlet
            roi: Return on Investment percentage
            expectancy: Expectancy metric
            quality_ratio: Ratio of quality trades to total trades
            win_ratio: Ratio of winning trades to total trades
            total_trades: Total trades

        Returns:
            True if agent was promoted, False otherwise
        """
        if not self.should_promote(gauntlet_score):
            return False

        with self._lock:
            print(f"\n{'='*60}")
            print("Global Hall of Fame - Promotion Routine")
            print(f"{'='*60}")
            print(f"🏆 Agent qualified with Gauntlet Score: {gauntlet_score:.2f}")

            # ATOMIC UPDATE: Re-download to get latest state
            print("⏳ Re-downloading global50.json for atomic update...")
            success = self._download_global_ledger()
            if not success:
                print("⚠ Failed to download latest global50.json. Aborting promotion.")
                print(f"{'='*60}\n")
                return False

            # Reload entries
            self._load_local_ledger()

            # Create new entry
            new_entry = GlobalHoFEntry(
                agent_id=agent.agent_id,
                run_name=self.run_name,
                gauntlet_score=gauntlet_score,
                generation=generation,
                roi=roi,
                expectancy=expectancy,
                quality_ratio=quality_ratio,
                win_ratio=win_ratio,
                total_trades=total_trades
            )

            # Merge: Add new agent
            self.entries.append(new_entry)

            # Sort: Rank by Gauntlet Score (descending)
            self.entries.sort(key=lambda e: e.gauntlet_score, reverse=True)

            # Cut: Identify dropouts
            dropouts = []
            if len(self.entries) > self.CAPACITY:
                dropouts = self.entries[self.CAPACITY:]
                self.entries = self.entries[:self.CAPACITY]

            # Sync Files
            self._sync_agent_files(new_entry, agent, dropouts)

            # Update Ledger
            self._save_local_ledger()
            self._upload_global_ledger()

            # Update Local Threshold
            self._update_entry_threshold()

            # Report
            new_rank = self.entries.index(new_entry) + 1
            print(f"✓ Agent promoted to Global 50!")
            print(f"  Rank: #{new_rank}")
            print(f"  Score: {gauntlet_score:.2f}")
            print(f"  Run: {self.run_name}")
            print(f"  New Threshold: {self.entry_threshold:.2f}")

            if dropouts:
                for dropout in dropouts:
                    print(f"  ⤵ Retired: {dropout.run_name} (Score: {dropout.gauntlet_score:.2f})")

            print(f"{'='*60}\n")

            return True

    def _sync_agent_files(self, new_entry: GlobalHoFEntry, agent: DDPGAgent,
                          dropouts: List[GlobalHoFEntry]):
        """
        Sync agent weight files between local and cloud.

        - Upload new agent's .pth to global50/agents/
        - Move dropout agents from agents/ to archive/ (both locally and on cloud)
        - Save dropout metadata as JSON scoresheets in archive/
        """
        # Upload new agent
        local_agent_path = self.local_agents_dir / new_entry.get_filename()
        agent.save(str(local_agent_path))

        cloud_agent_path = f"{self.cloud_base}/agents/{new_entry.get_filename()}"
        self.cloud_sync.upload_file(str(local_agent_path), cloud_agent_path, background=False)
        print(f"  ✓ Uploaded: {new_entry.get_filename()}")

        # Handle dropouts
        for dropout in dropouts:
            filename = dropout.get_filename()

            # Local: Move from agents/ to archive/
            local_src = self.local_agents_dir / filename
            local_dst = self.local_archive_dir / filename

            if local_src.exists():
                shutil.move(str(local_src), str(local_dst))
            else:
                # Download from cloud if not in local cache
                cloud_src = f"{self.cloud_base}/agents/{filename}"
                self.cloud_sync.download_file(cloud_src, str(local_dst))

            # Save metadata scoresheet
            scoresheet_filename = filename.replace('.pth', '.json')
            scoresheet_path = self.local_archive_dir / scoresheet_filename
            with open(scoresheet_path, 'w') as f:
                json.dump(dropout.to_dict(), f, indent=2)

            # Cloud: Upload to archive/ and delete from agents/
            cloud_archive_pth = f"{self.cloud_base}/archive/{filename}"
            cloud_archive_json = f"{self.cloud_base}/archive/{scoresheet_filename}"

            if local_dst.exists():
                self.cloud_sync.upload_file(str(local_dst), cloud_archive_pth, background=False)
            self.cloud_sync.upload_file(str(scoresheet_path), cloud_archive_json, background=False)

            # Delete from cloud agents/ (handled by next sync or manually)
            # Note: Most cloud APIs require explicit delete, but we can let it accumulate
            # or clean up in a separate maintenance script

    def get_random_agent(self) -> Optional[DDPGAgent]:
        """
        Load a random agent from the Global 50.

        This is used for diversity injection during plateau detection.
        Downloads the agent file from cloud storage if not in local cache.

        If our league is empty but fallback leagues exist, we download from those instead.

        Returns:
            Random DDPGAgent from Global 50, or None if Global 50 is empty/disabled
        """
        if not self.enabled:
            return None

        import random

        # Try our own league first
        if len(self.entries) > 0:
            random_entry = random.choice(self.entries)
            filename = random_entry.get_filename()
            local_agent_path = self.local_agents_dir / filename
            cloud_agent_path = f"{self.cloud_base}/agents/{filename}"

            # Download from cloud if not in cache
            if not local_agent_path.exists():
                success = self.cloud_sync.download_file(cloud_agent_path, str(local_agent_path))
                if not success:
                    print(f"⚠ Failed to download Global 50 agent: {filename}")
                    return None

            # Load agent
            try:
                agent = DDPGAgent(agent_id=-1)  # Temporary ID, will be reassigned
                agent.load(str(local_agent_path))
                return agent
            except Exception as e:
                print(f"⚠ Failed to load Global 50 agent: {e}")
                return None

        # Our league is empty - try fallback leagues
        if len(self.fallback_leagues) > 0:
            print(f"⚠ Current league ({self.league_rules.context_window_days} days) is empty.")
            print(f"  Attempting diversity injection from fallback league...")

            # Pick a random fallback league
            fallback = random.choice(self.fallback_leagues)
            fallback_id = fallback['context_window_id']
            fallback_cloud_base = fallback['cloud_base']

            # Download the fallback league's JSON to see available agents
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                temp_json_path = tmp.name

            try:
                cloud_json_path = f"{fallback_cloud_base}/global50.json"
                success = self.cloud_sync.download_file(cloud_json_path, temp_json_path)
                if not success:
                    print(f"  ✗ Failed to download fallback league JSON")
                    return None

                # Load entries
                with open(temp_json_path, 'r') as f:
                    data = json.load(f)
                fallback_entries = [GlobalHoFEntry.from_dict(e) for e in data.get('entries', [])]

                if len(fallback_entries) == 0:
                    print(f"  ✗ Fallback league is empty")
                    return None

                # Pick random agent from fallback league
                random_entry = random.choice(fallback_entries)
                filename = random_entry.get_filename()

                # Create local directory for fallback agents
                # Handle legacy structure (no subdirectory)
                if fallback.get('is_legacy', False):
                    fallback_local_dir = self.LOCAL_BASE_DIR / "agents"
                else:
                    fallback_local_dir = self.LOCAL_BASE_DIR / fallback_id / "agents"
                fallback_local_dir.mkdir(parents=True, exist_ok=True)
                local_agent_path = fallback_local_dir / filename

                # Download from fallback league's cloud storage
                cloud_agent_path = f"{fallback_cloud_base}/agents/{filename}"
                if not local_agent_path.exists():
                    success = self.cloud_sync.download_file(cloud_agent_path, str(local_agent_path))
                    if not success:
                        print(f"  ✗ Failed to download fallback agent: {filename}")
                        return None

                # Load agent
                agent = DDPGAgent(agent_id=-1)
                agent.load(str(local_agent_path))
                print(f"  ✓ Loaded agent from {fallback['context_window_days']} days league: {filename}")
                return agent

            except Exception as e:
                print(f"  ✗ Failed to load fallback agent: {e}")
                return None
            finally:
                # Clean up temp file
                import os
                if os.path.exists(temp_json_path):
                    os.unlink(temp_json_path)

        # No agents available anywhere
        return None

    def get_stats(self) -> Dict:
        """
        Get Global Hall of Fame statistics.

        Returns:
            Dictionary with Global HoF statistics
        """
        if not self.enabled or len(self.entries) == 0:
            return {
                'enabled': self.enabled,
                'size': 0,
                'entry_threshold': float('-inf'),
                'best_score': 0.0,
                'worst_score': 0.0,
                'mean_score': 0.0,
            }

        scores = [e.gauntlet_score for e in self.entries]

        return {
            'enabled': self.enabled,
            'size': len(self.entries),
            'entry_threshold': self.entry_threshold,
            'best_score': max(scores),
            'worst_score': min(scores),
            'mean_score': sum(scores) / len(scores),
            'league_compatible': self.league_compatible,
        }
