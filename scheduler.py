import os
import sys
from datetime import datetime


def run_scheduled(run_bot_fn) -> None:
    """Start the APScheduler blocking scheduler to run run_bot_fn daily."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        print("[scheduler] APScheduler not installed. Run: pip install APScheduler", file=sys.stderr)
        sys.exit(1)

    run_time = os.getenv("RUN_TIME", "10:00")
    try:
        hour, minute = map(int, run_time.split(":"))
    except ValueError:
        print(f"[scheduler] Invalid RUN_TIME '{run_time}', using 10:00 ET", file=sys.stderr)
        hour, minute = 10, 0

    scheduler = BlockingScheduler(timezone="America/New_York")
    trigger = CronTrigger(hour=hour, minute=minute, timezone="America/New_York")

    scheduler.add_job(run_bot_fn, trigger, id="nba_parlay_bot", name="NBA Parlay Bot")

    next_run = scheduler.get_jobs()[0].next_run_time
    print(f"[scheduler] NBA Parlay Bot scheduled daily at {hour:02d}:{minute:02d} ET")
    if next_run:
        print(f"[scheduler] Next run: {next_run.strftime('%A, %B %d %Y at %I:%M %p %Z')}")
    print("[scheduler] Press Ctrl+C to stop.\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n[scheduler] Stopped.")
