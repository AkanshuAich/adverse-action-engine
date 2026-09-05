"""Streamlit Community Cloud entry point.

A shim onto the package, not a second implementation. Community Cloud finds a
dependency file next to the entrypoint before it looks at the repository root,
so this directory is also where ``requirements.txt`` lives - which keeps the
root free of one, as the project's standards require.

That ordering is load-bearing rather than tidy. Community Cloud recognises a
root ``pyproject.toml`` and hands it to **poetry**; this project builds with
hatchling, so a root-level resolution would fail. Placing the requirements file
here means it is found first and poetry is never involved.
"""

from aae.console.app import main

main()
