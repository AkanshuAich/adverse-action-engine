"""Hugging Face Spaces entry point.

Spaces runs `app.py` at the repository root, so this is a shim rather than a
second implementation: the console itself lives in the package and is deployed
unchanged. A Space with its own copy of the UI would drift from the tested one.
"""

from aae.console.app import main

main()
