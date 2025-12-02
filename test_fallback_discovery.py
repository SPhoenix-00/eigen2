"""Test script to verify dynamic fallback league discovery."""

from utils.cloud_sync import get_cloud_sync_from_env
from utils.config import Config
from erl.league_rules import LeagueRules
from erl.global_hof import GlobalHallOfFame

# Initialize cloud sync
cloud_sync = get_cloud_sync_from_env()

print(f"Cloud Provider: {cloud_sync.provider}")
print(f"Bucket: {cloud_sync.bucket_name}")
print()

# Create a global HOF instance
league_rules = LeagueRules(context_window_days=Config.CONTEXT_WINDOW_DAYS)
print(f"Current context window: {league_rules.context_window_days} days")
print()

# Create Global HOF (this will trigger fallback discovery)
global_hof = GlobalHallOfFame(
    league_rules=league_rules,
    cloud_sync=cloud_sync
)

print()
print("=" * 60)
print("Discovered Fallback Leagues:")
print("=" * 60)
for league in global_hof.fallback_leagues:
    print(f"  - {league['context_window_days']} days ({league['entry_count']} agents)")
    print(f"    ID: {league['context_window_id']}")
    print(f"    Cloud base: {league['cloud_base']}")
    if league.get('is_legacy'):
        print(f"    Note: Legacy structure")
    print()

print(f"Total: {len(global_hof.fallback_leagues)} fallback leagues")
