"""Project-local command for running the complete test suite."""

from __future__ import annotations

import unittest
from pathlib import Path


def main() -> None:
    tests_directory = Path(__file__).resolve().parents[1] / "tests"
    suite = unittest.defaultTestLoader.discover(str(tests_directory))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
