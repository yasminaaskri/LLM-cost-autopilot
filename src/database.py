"""
SQLite database — schema init and all logging functions.
Every request, routing failure, escalation, and retrain event
is recorded here. The cost_if_highest_quality column is the foundation
of the savings headline metric.
"""

from __future__ import annotations
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.models import LLMResponse, ClassifierResult, QualityScore, EscalationResult

DB_PATH = Path(__file__).parent.parent / "data" / "autopilot.db"


def get_connection() -> sqlite3.Connection:
    """Return a connection with row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads + writes
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """
    Create all tables if they don't exist.
    Safe to call on every startup — idempotent.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.executescript("""
        -- ── Main request log ─────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS requests (
            id                TEXT PRIMARY KEY,
            timestamp         TEXT NOT NULL,
            user_id           TEXT DEFAULT 'default',

            -- Prompt metadata
            prompt_hash       TEXT NOT NULL,
            prompt_preview    TEXT,           -- first 150 chars

            -- Classifier output
            complexity_tier   INTEGER,        -- 1 | 2 | 3
            classifier_conf   REAL,           -- 0.0 – 1.0
            low_confidence    INTEGER DEFAULT 0,  -- bool

            -- Routing decision
            routed_model      TEXT NOT NULL,
            provider          TEXT NOT NULL,

            -- Token counts
            input_tokens      INTEGER,
            output_tokens     INTEGER,

            -- Cost (the core metric)
            cost_usd          REAL,
            cost_if_highest_quality REAL NOT NULL,  -- ALWAYS populated, never NULL

            -- Performance
            latency_ms        REAL,

            -- Async verifier output (populated after user gets response)
            quality_score     REAL,           -- NULL until verifier runs
            verified_at       TEXT,

            -- Escalation
            escalated         INTEGER DEFAULT 0,  -- bool
            escalated_model   TEXT,
            cost_delta_usd    REAL DEFAULT 0.0,

            -- Output preview
            output_preview    TEXT            -- first 200 chars
        );

        CREATE INDEX IF NOT EXISTS idx_requests_timestamp
            ON requests(timestamp);
        CREATE INDEX IF NOT EXISTS idx_requests_tier
            ON requests(complexity_tier);
        CREATE INDEX IF NOT EXISTS idx_requests_model
            ON requests(routed_model);

        -- ── Routing failures (flywheel fuel) ──────────────────────────────────
        CREATE TABLE IF NOT EXISTS routing_failures (
            id                TEXT PRIMARY KEY,
            request_id        TEXT REFERENCES requests(id),
            timestamp         TEXT NOT NULL,
            prompt            TEXT NOT NULL,       -- full prompt for retraining
            classified_tier   INTEGER,
            correct_tier      INTEGER,             -- what it should have been
            quality_gap       REAL,
            failure_reason    TEXT,
            used_in_retrain   INTEGER DEFAULT 0    -- bool: 0 until retrain.py runs
        );

        -- ── Retrain audit log ─────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS retrain_log (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT NOT NULL,
            old_accuracy      REAL,
            new_accuracy      REAL,
            num_original      INTEGER,
            num_failures      INTEGER,
            total_examples    INTEGER,
            model_replaced    INTEGER,             -- bool
            model_hash        TEXT
        );

        -- ── Baseline comparison runs (from scripts/baseline_run.py) ───────────
        CREATE TABLE IF NOT EXISTS baseline_runs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp     TEXT NOT NULL,
            prompt_index      INTEGER,
            prompt_preview    TEXT,
            model_key         TEXT,
            output_preview    TEXT,
            input_tokens      INTEGER,
            output_tokens     INTEGER,
            cost_usd          REAL,
            latency_ms        REAL
        );
        """)

    print(f"✓ Database ready at {DB_PATH}")


# ── Logging functions ──────────────────────────────────────────────────────────

def log_request(
    response: LLMResponse,
    classifier: Optional[ClassifierResult],
    user_id: str = "default",
    prompt_preview: str = "",
    output_preview: str = "",
) -> str:
    """
    Insert a completed request into the requests table.
    Returns the request_id (UUID) for use by the async verifier.

    cost_if_highest_quality MUST be set on response before calling this.
    """
    prompt_hash = hashlib.sha256(prompt_preview.encode()).hexdigest()[:16]

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO requests (
                id, timestamp, user_id,
                prompt_hash, prompt_preview,
                complexity_tier, classifier_conf, low_confidence,
                routed_model, provider,
                input_tokens, output_tokens,
                cost_usd, cost_if_highest_quality,
                latency_ms,
                output_preview
            ) VALUES (
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?,
                ?,
                ?
            )
        """, (
            response.request_id,
            response.timestamp.isoformat(),
            user_id,
            prompt_hash,
            prompt_preview[:150],
            classifier.tier if classifier else None,
            classifier.confidence if classifier else None,
            int(classifier.low_confidence) if classifier else 0,
            response.model_id,
            response.provider,
            response.input_tokens,
            response.output_tokens,
            response.cost_usd,
            response.cost_if_highest_quality,   # never None — enforced by LLMResponse
            response.latency_ms,
            output_preview[:200],
        ))

    return response.request_id


def update_quality_score(request_id: str,
                         score: QualityScore) -> None:
    """Called by the async verifier after judging is complete."""
    with get_connection() as conn:
        conn.execute("""
            UPDATE requests
            SET quality_score = ?,
                verified_at   = ?
            WHERE id = ?
        """, (
            score.cheap_score,
            datetime.utcnow().isoformat(),
            request_id,
        ))


def update_escalation(request_id: str,
                      result: EscalationResult) -> None:
    """Called by the escalation engine if it fires."""
    with get_connection() as conn:
        conn.execute("""
            UPDATE requests
            SET escalated       = ?,
                escalated_model = ?,
                cost_delta_usd  = ?
            WHERE id = ?
        """, (
            int(result.escalated),
            result.escalated_model,
            result.cost_delta_usd,
            request_id,
        ))


def log_routing_failure(request_id: str,
                        prompt: str,
                        classified_tier: int,
                        quality_score: QualityScore) -> None:
    """
    Record a routing failure for the flywheel.
    The correct_tier is inferred: if cheap model scored < 3 on a
    task that expensive model scored 4+, correct tier is one level up.
    """
    import uuid as _uuid
    # Simple heuristic: if gap > 2, bump up two tiers; else bump one
    gap = quality_score.quality_gap
    correct_tier = min(3, classified_tier + (2 if gap > 2.5 else 1))

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO routing_failures (
                id, request_id, timestamp,
                prompt, classified_tier, correct_tier,
                quality_gap, failure_reason, used_in_retrain
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            str(_uuid.uuid4()),
            request_id,
            datetime.utcnow().isoformat(),
            prompt,
            classified_tier,
            correct_tier,
            quality_score.quality_gap,
            quality_score.failure_reason,
        ))


def log_baseline_run(run_timestamp: str,
                     prompt_index: int,
                     prompt_preview: str,
                     model_key: str,
                     output_preview: str,
                     input_tokens: int,
                     output_tokens: int,
                     cost_usd: float,
                     latency_ms: float) -> None:
    """Log a single row from scripts/baseline_run.py."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO baseline_runs (
                run_timestamp, prompt_index, prompt_preview,
                model_key, output_preview,
                input_tokens, output_tokens,
                cost_usd, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_timestamp, prompt_index, prompt_preview[:150],
            model_key, output_preview[:200],
            input_tokens, output_tokens,
            cost_usd, latency_ms,
        ))


# ── Read functions (used by dashboard + /v1/stats) ────────────────────────────

def get_summary_stats() -> dict:
    """
    Aggregate stats for GET /v1/stats and the dashboard headline cards.
    Returns the savings_pct — the project's headline metric.
    """
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                                        AS total_requests,
                ROUND(SUM(cost_usd), 6)                        AS total_cost_usd,
                ROUND(SUM(cost_if_highest_quality), 6)         AS total_baseline_cost,
                ROUND(SUM(cost_if_highest_quality) - SUM(cost_usd), 6) AS savings_usd,
                ROUND(
                    (1.0 - SUM(cost_usd) / NULLIF(SUM(cost_if_highest_quality), 0)) * 100,
                    1
                )                                               AS savings_pct,
                ROUND(AVG(quality_score), 2)                    AS avg_quality_score,
                ROUND(AVG(latency_ms), 0)                       AS avg_latency_ms
            FROM requests
        """).fetchone()

        escalation_row = conn.execute("""
            SELECT ROUND(AVG(escalated) * 100, 1) AS escalation_rate_pct
            FROM requests
            WHERE timestamp > datetime('now', '-7 days')
        """).fetchone()

        tier_rows = conn.execute("""
            SELECT complexity_tier, COUNT(*) AS n
            FROM requests
            WHERE complexity_tier IS NOT NULL
            GROUP BY complexity_tier
        """).fetchall()

        model_rows = conn.execute("""
            SELECT routed_model, COUNT(*) AS n,
                   ROUND(AVG(quality_score), 2) AS avg_quality
            FROM requests
            GROUP BY routed_model
            ORDER BY n DESC
        """).fetchall()

    return {
        "total_requests":      row["total_requests"] or 0,
        "total_cost_usd":      row["total_cost_usd"] or 0.0,
        "total_baseline_cost": row["total_baseline_cost"] or 0.0,
        "savings_usd":         row["savings_usd"] or 0.0,
        "savings_pct":         row["savings_pct"] or 0.0,
        "avg_quality_score":   row["avg_quality_score"],
        "avg_latency_ms":      row["avg_latency_ms"] or 0,
        "escalation_rate_pct": escalation_row["escalation_rate_pct"] or 0.0,
        "requests_by_tier":    {str(r["complexity_tier"]): r["n"] for r in tier_rows},
        "requests_by_model":   {r["routed_model"]: {"count": r["n"], "avg_quality": r["avg_quality"]} for r in model_rows},
    }


def get_recent_requests(limit: int = 50) -> list[dict]:
    """Fetch the last N requests for the dashboard audit table."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, timestamp, prompt_preview, complexity_tier,
                   classifier_conf, routed_model, cost_usd,
                   cost_if_highest_quality, latency_ms, quality_score,
                   escalated, escalated_model, output_preview
            FROM requests
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [dict(r) for r in rows]


def get_cost_timeseries(days: int = 7) -> list[dict]:
    """Daily cost actual vs baseline — for the dashboard bar chart."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                DATE(timestamp)                AS day,
                ROUND(SUM(cost_usd), 6)        AS cost_actual,
                ROUND(SUM(cost_if_highest_quality), 6)   AS cost_baseline,
                COUNT(*)                        AS num_requests
            FROM requests
            WHERE timestamp > datetime('now', ? || ' days')
            GROUP BY day
            ORDER BY day
        """, (f"-{days}",)).fetchall()

    return [dict(r) for r in rows]


def export_failures_for_retrain() -> list[dict]:
    """
    Return all unprocessed routing failures as (prompt, correct_tier) pairs.
    Marks them as used_in_retrain=1 so they're not exported twice.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, prompt, correct_tier
            FROM routing_failures
            WHERE used_in_retrain = 0
        """).fetchall()

        if rows:
            ids = [r["id"] for r in rows]
            placeholders = ",".join("?" * len(ids))
            conn.execute(f"""
                UPDATE routing_failures
                SET used_in_retrain = 1
                WHERE id IN ({placeholders})
            """, ids)

    return [{"prompt": r["prompt"], "tier": r["correct_tier"]} for r in rows]