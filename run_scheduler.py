"""
Entry point for the prediction scheduler service.
Handles signal handling and starts the scheduler.
"""

import signal
import sys
from pathlib import Path

# Add scripts to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from scripts.scheduler import PredictionScheduler


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    print("\nReceived shutdown signal, stopping scheduler...")
    sys.exit(0)


def main():
    """Main entry point."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start scheduler
    scheduler = PredictionScheduler()
    scheduler.start()


if __name__ == "__main__":
    main()

