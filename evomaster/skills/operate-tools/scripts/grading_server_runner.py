import argparse
import logging
from pathlib import Path

from werkzeug.serving import make_server

from search_dataset_tools.operate_submission._submission_utils import _create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run standalone grading server")
    parser.add_argument("--data-root", required=True, help="Path to ML_MASTER_DATA_ROOT")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=5003, help="Bind port")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data root does not exist: {data_root}")

    app = _create_app(data_root)

    logger.info("Starting standalone grading server on http://%s:%s", args.host, args.port)
    logger.info("Using data root: %s", data_root)

    httpd = make_server(args.host, args.port, app)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down grading server")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()