"""
scripts/worker.py — background worker process.

Entry point for the 'worker' Docker service. Runs two things:
  1. Weekly retrain cron (Sunday 02:00 UTC)
  2. Periodic escalation rate check + alert logging (every hour)

This is a separate process from the API so it never competes for
request-handling capacity. It shares the same SQLite DB and models/
directory via Docker named volumes.

Run manually:   python scripts/worker.py
Docker service: command: python scripts/worker.py
"""

from __future__ import annotations
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import schedule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("worker")


# ── Job: weekly retrain ────────────────────────────────────────────────────────

def job_retrain() -> None:
    """
    Retrain the classifier from accumulated routing failures.
    Runs every Sunday at 02:00 UTC.
    Only saves a new model if accuracy doesn't regress > 2%.
    """
    logger.info("=== SCHEDULED RETRAIN starting ===")
    try:
        from scripts.retrain import retrain
        result = retrain()

        if result.get("status") == "replaced":
            logger.info(
                "Retrain SUCCESS: accuracy %s → %s  (%d new failure examples absorbed)",
                f"{result['old_accuracy']:.2%}",
                f"{result['new_accuracy']:.2%}",
                result["n_failures"],
            )
        elif result.get("status") == "skipped":
            logger.info("Retrain SKIPPED: no new failure examples since last run")
        elif result.get("status") == "rejected_regression":
            logger.warning(
                "Retrain REJECTED: new accuracy %.2%% < old %.2%% − 2%%. Keeping old model.",
                result["new_accuracy"],
                result["old_accuracy"],
            )
        else:
            logger.info("Retrain result: %s", result)

    except Exception as e:
        logger.error("Retrain job FAILED: %s", e, exc_info=True)

    logger.info("=== SCHEDULED RETRAIN done ===")


# ── Job: escalation rate check ─────────────────────────────────────────────────

def job_check_escalation_rate() -> None:
    """
    Check escalation rate over the last 100 requests.
    Logs a WARNING if > 20% — signal that the classifier needs attention.
    Runs every hour.
    """
    try:
        from src.database import get_connection
        with get_connection() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                    AS n,
                    ROUND(AVG(escalated)*100,1) AS rate_pct
                FROM (
                    SELECT escalated FROM requests
                    ORDER BY timestamp DESC
                    LIMIT 100
                )
            """).fetchone()

        if row and row["n"] >= 10:
            rate = row["rate_pct"] or 0.0
            if rate > 20.0:
                logger.warning(
                    "ESCALATION RATE ALERT: %.1f%% over last %d requests "
                    "(threshold 20%%). Consider triggering retrain via "
                    "POST /v1/admin/retrain or checking classifier accuracy.",
                    rate, row["n"],
                )
            else:
                logger.info(
                    "Escalation rate check OK: %.1f%% over last %d requests",
                    rate, row["n"],
                )
    except Exception as e:
        logger.error("Escalation rate check FAILED: %s", e)


# ── Job: pending failure count log ────────────────────────────────────────────

def job_log_pending_failures() -> None:
    """Log how many routing failures are waiting for the next retrain."""
    try:
        from src.database import get_connection
        with get_connection() as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM routing_failures WHERE used_in_retrain=0"
            ).fetchone()["n"]
        if n > 0:
            logger.info("Pending routing failures (unused in retrain): %d", n)
    except Exception as e:
        logger.error("Pending failure count check FAILED: %s", e)


# ── Schedule setup ─────────────────────────────────────────────────────────────

def setup_schedule() -> None:
    # Retrain: weekly, Sunday at 02:00 UTC
    schedule.every().sunday.at("02:00").do(job_retrain)

    # Escalation rate check: every hour
    schedule.every(1).hours.do(job_check_escalation_rate)

    # Pending failure log: every 6 hours
    schedule.every(6).hours.do(job_log_pending_failures)

    logger.info("Schedule configured:")
    logger.info("  - Retrain:              Sundays at 02:00 UTC")
    logger.info("  - Escalation check:     every hour")
    logger.info("  - Pending failure log:  every 6 hours")


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Worker starting — pid=%d", __import__("os").getpid())

    # Run startup checks immediately (non-blocking)
    job_check_escalation_rate()
    job_log_pending_failures()

    setup_schedule()

    logger.info("Entering schedule loop. Ctrl-C to stop.")
    while True:
        try:
            schedule.run_pending()
            time.sleep(30)   # check schedule every 30 s
        except KeyboardInterrupt:
            logger.info("Worker stopped by keyboard interrupt")
            break
        except Exception as e:
            logger.error("Unexpected error in schedule loop: %s", e, exc_info=True)
            time.sleep(60)   # back off before retrying


if __name__ == "__main__":
    main()
