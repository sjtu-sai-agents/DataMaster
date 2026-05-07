"""CLI entry point

python -m evomaster.interface.feishu [--config PATH] [--agent NAME]
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path


def main() -> int:
    """Parse CLI arguments and start the Feishu bot.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="EvoMaster Feishu Bot — receive Feishu messages and execute playground tasks",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the Feishu Bot config file (default: configs/feishu/config.yaml)",
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Override the default agent name",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)

    # Determine project root directory
    # __main__.py is located at evomaster/interface/feishu/, project_root is 3 levels up
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    # Load configuration
    config_path = Path(args.config) if args.config else project_root / "configs" / "feishu" / "config.yaml"

    from .config import load_feishu_config

    try:
        config = load_feishu_config(config_path, project_root=project_root)
    except FileNotFoundError as e:
        logger.error("配置文件未找到: %s", e)
        return 1

    # Command-line override
    if args.agent:
        config.default_agent = args.agent

    # Create the Bot
    from .app import FeishuBot

    bot = FeishuBot(config=config, project_root=project_root)

    # Signal handling
    def _shutdown(signum, _frame):
        """Handle termination signals by stopping the bot."""
        logger.info("Received signal %s, shutting down...", signum)
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start (blocking)
    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
