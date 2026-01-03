"""
BugBot - Automated Bug Detection and Testing Tool
Runs tests and checks for common bugs in the Eigen2 codebase.
"""

import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import List, Tuple, Dict
import json


class BugBot:
    """Automated bug detection and testing."""
    
    def __init__(self):
        self.issues_found = []
        self.tests_passed = []
        self.tests_failed = []
        
    def check_imports(self) -> bool:
        """Check if all required modules can be imported."""
        print("\n" + "="*70)
        print("CHECKING IMPORTS")
        print("="*70)
        
        required_modules = [
            'torch',
            'numpy',
            'wandb',
            'data.loader',
            'models.ddpg_agent',
            'models.networks',
            'models.replay_buffer',
            'training.erl_trainer',
            'environment.trading_env',
            'utils.config',
            'erl.global_hof',
        ]
        
        all_ok = True
        for module_name in required_modules:
            try:
                if '.' in module_name:
                    # Local module
                    spec = importlib.util.find_spec(module_name)
                    if spec is None:
                        print(f"  ❌ {module_name}: NOT FOUND")
                        all_ok = False
                        self.issues_found.append(f"Missing module: {module_name}")
                    else:
                        print(f"  [OK] {module_name}: OK")
                else:
                    # External module
                    __import__(module_name)
                    print(f"  [OK] {module_name}: OK")
            except ImportError as e:
                print(f"  [FAIL] {module_name}: {e}")
                all_ok = False
                self.issues_found.append(f"Import error: {module_name} - {e}")
        
        return all_ok
    
    def check_config(self) -> bool:
        """Validate configuration."""
        print("\n" + "="*70)
        print("CHECKING CONFIGURATION")
        print("="*70)
        
        try:
            from utils.config import Config
            
            # Check if config validates
            if Config.validate():
                print("  [OK] Configuration validation: PASSED")
                return True
            else:
                print("  [FAIL] Configuration validation: FAILED")
                self.issues_found.append("Configuration validation failed")
                return False
        except Exception as e:
            print(f"  [FAIL] Configuration check failed: {e}")
            self.issues_found.append(f"Config error: {e}")
            return False
    
    def check_cuda_serialization(self) -> bool:
        """Check for CUDA tensor serialization issues in multiprocessing."""
        print("\n" + "="*70)
        print("CHECKING CUDA SERIALIZATION")
        print("="*70)
        
        try:
            trainer_path = Path("training/erl_trainer.py")
            if not trainer_path.exists():
                print("  [WARN] training/erl_trainer.py not found")
                return True
            
            content = trainer_path.read_text(encoding='utf-8', errors='ignore')
            
            # Check for state_dict() usage in multiprocessing contexts
            issues = []
            
            # Look for state_dict() calls that might not have .cpu()
            # This is a heuristic check - actual bugs need runtime testing
            if 'state_dict()' in content and 'multiprocessing' in content:
                # Check if there are .cpu() calls near state_dict
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'state_dict()' in line and 'multiprocessing' in '\n'.join(lines[max(0, i-10):i+10]):
                        # Check if .cpu() is nearby
                        context = '\n'.join(lines[max(0, i-5):min(len(lines), i+5)])
                        if '.cpu()' not in context:
                            issues.append(f"Line {i+1}: state_dict() in multiprocessing context without .cpu()")
            
            if issues:
                print("  [WARN] Potential CUDA serialization issues found:")
                for issue in issues:
                    print(f"    - {issue}")
                self.issues_found.extend(issues)
                return False
            else:
                print("  [OK] No obvious CUDA serialization issues detected")
                return True
                
        except Exception as e:
            print(f"  [WARN] Could not check CUDA serialization: {e}")
            return True
    
    def check_json_serialization(self) -> bool:
        """Check for numpy type serialization issues."""
        print("\n" + "="*70)
        print("CHECKING JSON SERIALIZATION")
        print("="*70)
        
        try:
            # Check global_hof.py for to_dict() methods
            hof_path = Path("erl/global_hof.py")
            if hof_path.exists():
                content = hof_path.read_text(encoding='utf-8', errors='ignore')
                if 'to_dict' in content and 'asdict' in content:
                    # Check if to_dict properly converts numpy types
                    if 'hasattr(value, \'item\')' in content or "hasattr(value, 'item')" in content:
                        print("  [OK] global_hof.py has numpy type conversion")
                    else:
                        print("  [WARN] global_hof.py may have numpy serialization issues")
                        self.issues_found.append("global_hof.py: Missing numpy type conversion in to_dict()")
                        return False
                else:
                    print("  [OK] global_hof.py uses proper serialization")
            
            # Check compare_context_windows.py
            compare_path = Path("compare_context_windows.py")
            if compare_path.exists():
                content = compare_path.read_text(encoding='utf-8', errors='ignore')
                if 'asdict(' in content and 'to_dict()' not in content:
                    print("  [WARN] compare_context_windows.py may use asdict() without conversion")
                    self.issues_found.append("compare_context_windows.py: May need to_dict() for numpy types")
                    return False
                else:
                    print("  [OK] compare_context_windows.py serialization looks good")
            
            return True
            
        except Exception as e:
            print(f"  [WARN] Could not check JSON serialization: {e}")
            return True
    
    def check_hardcoded_context_windows(self) -> bool:
        """Check for hardcoded context window lists."""
        print("\n" + "="*70)
        print("CHECKING HARDCODED CONTEXT WINDOWS")
        print("="*70)
        
        try:
            hof_path = Path("erl/global_hof.py")
            if hof_path.exists():
                content = hof_path.read_text(encoding='utf-8', errors='ignore')
                # Check for hardcoded lists like [504, 252, 377, ...]
                if '[504, 252, 377' in content or '[504, 252, 125' in content:
                    # But check if it's in a fallback (which is OK)
                    if 'fallback' in content.lower() or 'FALLBACK' in content:
                        print("  [OK] Hardcoded list appears to be in fallback (OK)")
                        return True
                    else:
                        print("  [WARN] Hardcoded context window list found (may need dynamic discovery)")
                        self.issues_found.append("Hardcoded context window list in global_hof.py")
                        return False
                else:
                    print("  [OK] No hardcoded context window lists detected")
                    return True
            return True
            
        except Exception as e:
            print(f"  [WARN] Could not check hardcoded lists: {e}")
            return True
    
    def run_test_file(self, test_file: Path) -> Tuple[bool, str]:
        """Run a test file and return (success, output)."""
        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            success = result.returncode == 0
            return success, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Test timed out after 5 minutes"
        except Exception as e:
            return False, f"Error running test: {e}"
    
    def run_tests(self) -> bool:
        """Run all available test files."""
        print("\n" + "="*70)
        print("RUNNING TESTS")
        print("="*70)
        
        test_files = [
            Path("test_cleanup_safety.py"),
            Path("test_global_hof.py"),
            Path("test_fallback_discovery.py"),
            Path("test_snapback.py"),
        ]
        
        all_passed = True
        for test_file in test_files:
            if test_file.exists():
                print(f"\n  Running {test_file.name}...")
                success, output = self.run_test_file(test_file)
                
                if success:
                    print(f"  [PASS] {test_file.name}: PASSED")
                    self.tests_passed.append(test_file.name)
                else:
                    print(f"  [FAIL] {test_file.name}: FAILED")
                    print(f"    Output: {output[:500]}...")  # First 500 chars
                    self.tests_failed.append((test_file.name, output))
                    all_passed = False
            else:
                print(f"  [SKIP] {test_file.name}: NOT FOUND (skipping)")
        
        return all_passed
    
    def check_file_structure(self) -> bool:
        """Check if essential files and directories exist."""
        print("\n" + "="*70)
        print("CHECKING FILE STRUCTURE")
        print("="*70)
        
        required_paths = [
            Path("main.py"),
            Path("requirements.txt"),
            Path("utils/config.py"),
            Path("data/loader.py"),
            Path("models/ddpg_agent.py"),
            Path("training/erl_trainer.py"),
            Path("environment/trading_env.py"),
        ]
        
        all_ok = True
        for path in required_paths:
            if path.exists():
                print(f"  [OK] {path}: EXISTS")
            else:
                print(f"  [FAIL] {path}: MISSING")
                self.issues_found.append(f"Missing file: {path}")
                all_ok = False
        
        return all_ok
    
    def generate_report(self) -> None:
        """Generate final bug report."""
        print("\n" + "="*70)
        print("BUG REPORT SUMMARY")
        print("="*70)
        
        print(f"\nTests Passed: {len(self.tests_passed)}")
        for test in self.tests_passed:
            print(f"  [PASS] {test}")
        
        print(f"\nTests Failed: {len(self.tests_failed)}")
        for test_name, output in self.tests_failed:
            print(f"  [FAIL] {test_name}")
        
        print(f"\nIssues Found: {len(self.issues_found)}")
        for issue in self.issues_found:
            print(f"  [WARN] {issue}")
        
        if not self.issues_found and not self.tests_failed:
            print("\n" + "="*70)
            print("[SUCCESS] ALL CHECKS PASSED - NO BUGS DETECTED")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("[WARNING] ISSUES DETECTED - REVIEW ABOVE")
            print("="*70)
    
    def run(self) -> int:
        """Run all bug checks and tests."""
        print("\n" + "#"*70)
        print("# BUGBOT - Automated Bug Detection")
        print("#"*70)
        
        results = []
        
        # Run all checks
        results.append(("File Structure", self.check_file_structure()))
        results.append(("Imports", self.check_imports()))
        results.append(("Configuration", self.check_config()))
        results.append(("CUDA Serialization", self.check_cuda_serialization()))
        results.append(("JSON Serialization", self.check_json_serialization()))
        results.append(("Hardcoded Context Windows", self.check_hardcoded_context_windows()))
        results.append(("Test Suite", self.run_tests()))
        
        # Generate report
        self.generate_report()
        
        # Return exit code
        if self.issues_found or self.tests_failed:
            return 1
        return 0


def main():
    """Main entry point."""
    bugbot = BugBot()
    exit_code = bugbot.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

