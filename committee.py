"""
Backward-compatible proxy for the committee package.

The committee module has been refactored into a package (committee/).
This file preserves the `python committee.py` CLI invocation.

Preferred usage: python -m committee [--draft] [--validate] [--mirror] ...
"""
from committee.__main__ import main

if __name__ == "__main__":
    main()
