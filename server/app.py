"""Root-level OpenEnv server wrapper.

OpenEnv validation expects server/app.py in repository root.
This module re-exports the real AlertStorm app implementation.
"""

from alertstorm.server.app import app, main as _alertstorm_main


def main(host: str = "0.0.0.0", port: int = 8000):
    """Root-level server entrypoint expected by OpenEnv validate."""
    _alertstorm_main(host=host, port=port)


if __name__ == "__main__":
    main()
