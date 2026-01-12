"""
Scheduler Service for Football Match Prediction
Uses APScheduler to automatically schedule prediction jobs after matchday ends.
"""

import datetime
import os
import pickle
import smtplib
import subprocess
import sys
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.date import DateTrigger

# Add BASE_DIR and scripts to path for imports
BASE_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from data_loader import get_current_season

DATA_DIR = BASE_DIR / "data"


def send_email(
    subject: str,
    body: str,
    receiver: Optional[str] = None,
    smtp_server: Optional[str] = None,
    smtp_port: Optional[int] = None,
    smtp_user: Optional[str] = None,
    smtp_password: Optional[str] = None,
) -> bool:
    """
    Send email with logs.
    
    Args:
        subject: Email subject
        body: Email body (logs)
        receiver: Email receiver (from env if not provided)
        smtp_server: SMTP server (default: smtp.gmail.com)
        smtp_port: SMTP port (default: 587)
        smtp_user: SMTP username (optional)
        smtp_password: SMTP password (optional)
        
    Returns:
        bool: True if email sent successfully
    """
    try:
        import dotenv
        dotenv.load_dotenv()
        
        receiver = receiver or os.getenv("EMAIL_RECEIVER")
        smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        smtp_user = smtp_user or os.getenv("SMTP_USER")
        smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        
        if not receiver:
            print("ERROR: EMAIL_RECEIVER not set, skipping email send")
            return False
        
        # For Gmail and most SMTP servers, authentication is required
        if not smtp_user or not smtp_password:
            print(f"ERROR: SMTP_USER and SMTP_PASSWORD must be set for email sending.")
            print(f"  SMTP_SERVER: {smtp_server}")
            print(f"  SMTP_PORT: {smtp_port}")
            print(f"  SMTP_USER: {'set' if smtp_user else 'NOT SET'}")
            print(f"  SMTP_PASSWORD: {'set' if smtp_password else 'NOT SET'}")
            print(f"  Please set SMTP_USER and SMTP_PASSWORD in .env file")
            return False
        
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = receiver
        
        # Send email
        try:
            print(f"Attempting to send email to {receiver} via {smtp_server}:{smtp_port}...")
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                print("  Connecting to SMTP server...")
                server.starttls()
                print("  Starting TLS...")
                print(f"  Logging in as {smtp_user}...")
                server.login(smtp_user, smtp_password)
                print("  Sending message...")
                server.send_message(msg)
            
            print(f"✓ Email sent successfully to {receiver}")
            return True
        except smtplib.SMTPAuthenticationError as e:
            print(f"ERROR: SMTP Authentication failed: {e}")
            print("  Check your SMTP_USER and SMTP_PASSWORD in .env file")
            print("  For Gmail, you need to use an App Password, not your regular password")
            return False
        except smtplib.SMTPException as e:
            print(f"ERROR: SMTP error occurred: {e}")
            import traceback
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"ERROR: Unexpected error sending email: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"Error sending email: {e}")
        import traceback
        traceback.print_exc()
        return False


def detect_matchday_end(match_df: pd.DataFrame) -> Optional[datetime.datetime]:
    """
    Detect the start time of the last match in the last completed matchday.
    
    A matchday is considered completed if all matches are either finished or cancelled.
    This handles the edge case where matches are postponed/cancelled.
    
    Args:
        match_df: DataFrame with match data
        
    Returns:
        datetime of the latest match start time of the last completed matchday, or None
    """
    if len(match_df) == 0:
        return None
    
    # Group by matchday and season
    matchday_groups = match_df.groupby(["matchDay", "season"])
    
    completed_matchdays = []
    
    for (matchday, season), group in matchday_groups:
        # Check if all matches in this matchday are finished or cancelled
        # (not future anymore)
        all_completed = ~(group["status"] == "future").any()
        
        if all_completed:
            # Get the latest match start time in this matchday
            latest_match_start = group["date"].max()
            
            # Convert to datetime if needed
            if isinstance(latest_match_start, pd.Timestamp):
                latest_match_start = latest_match_start.to_pydatetime()
            elif not isinstance(latest_match_start, datetime.datetime):
                # If it's a date object, assume evening match at 20:30
                latest_match_start = datetime.datetime.combine(
                    latest_match_start, 
                    datetime.time(20, 30)
                )
            
            completed_matchdays.append({
                "matchday": matchday,
                "season": season,
                "end_date": latest_match_start,
            })
    
    if not completed_matchdays:
        return None
    
    # Get the latest completed matchday
    latest_matchday = max(completed_matchdays, key=lambda x: x["end_date"])
    return latest_matchday["end_date"]


def run_prediction_job() -> tuple[str, Optional[datetime.datetime]]:
    """
    Run prediction and upload job.
    
    Returns:
        tuple: (logs, next_job_datetime)
    """
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    logs = []
    logs.append("=" * 60)
    logs.append("Football Prediction Job")
    logs.append("=" * 60)
    logs.append(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logs.append("")
    
    # Run predict.py directly (no CSV saving)
    logs.append("Running predictions (no CSV save)...")
    logs.append("-" * 60)
    try:
        # Import and call directly to avoid CSV creation
        from scripts.predict import generate_predictions
        
        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            results_df = generate_predictions(save_csv=False, verbose=True)
        
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        logs.append(stdout_output)
        if stderr_output:
            logs.append("STDERR:")
            logs.append(stderr_output)
        
        # Also print to console for visibility
        print(stdout_output)
        if stderr_output:
            print("STDERR:", stderr_output)
            
    except Exception as e:
        error_msg = f"ERROR running predictions: {e}"
        logs.append(error_msg)
        print(error_msg)
        import traceback
        tb = traceback.format_exc()
        logs.append(tb)
        print(tb)
    
    logs.append("")
    logs.append("-" * 60)
    logs.append("Running upload_predictions.py --submit...")
    logs.append("-" * 60)
    
    # Run upload_predictions.py --submit
    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "scripts" / "upload_predictions.py"), "--submit"],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
        )
        logs.append(result.stdout)
        print(result.stdout)  # Print to console
        if result.stderr:
            logs.append("STDERR:")
            logs.append(result.stderr)
            print("STDERR:", result.stderr)  # Print to console
        if result.returncode != 0:
            warning = f"WARNING: upload_predictions.py exited with code {result.returncode}"
            logs.append(warning)
            print(warning)
    except Exception as e:
        error_msg = f"ERROR running upload_predictions.py: {e}"
        logs.append(error_msg)
        print(error_msg)
        import traceback
        tb = traceback.format_exc()
        logs.append(tb)
        print(tb)
    
    logs.append("")
    logs.append("=" * 60)
    logs.append(f"Job completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logs.append("=" * 60)
    
    # Determine next job time
    next_job_datetime = schedule_next_prediction()
    
    log_text = "\n".join(logs)
    return log_text, next_job_datetime


def schedule_next_prediction() -> Optional[datetime.datetime]:
    """
    Schedule the next prediction job based on the matchday we just predicted for.
    
    Uses next_matchday_df to find the latest match start time of the matchday,
    then schedules the next job for 3 hours after that match begins.
    
    Returns:
        datetime of next scheduled job, or None if not scheduled
    """
    current_date = datetime.datetime.now()
    
    try:
        # Load next_matchday_df to get the matchday we just predicted for
        next_matchday_path = DATA_DIR / "next_matchday_df.pck"
        
        if not next_matchday_path.exists():
            print(f"next_matchday_df.pck not found, trying to use matchday end detection")
            # Fallback: use matchday end detection
            current_season = get_current_season(current_date)
            pickle_file = DATA_DIR / f"match_df_{current_season}.pck"
            
            if not pickle_file.exists():
                print(f"No pickle file found for season {current_season}")
                return None
            
            match_df = pd.DataFrame(pickle.load(open(pickle_file, "rb")))
            matchday_end_date = detect_matchday_end(match_df)
            
            if matchday_end_date is None:
                print("No completed matchday found")
                return None
            
            # Schedule for matchday_end + 3 hours
            if isinstance(matchday_end_date, datetime.datetime):
                next_job_datetime = matchday_end_date + datetime.timedelta(hours=3)
            else:
                # If it's just a date, assume evening match (20:30) + 3 hours
                next_job_datetime = datetime.datetime.combine(
                    matchday_end_date, 
                    datetime.time(23, 30)
                )
        else:
            next_matchday_df = pd.DataFrame(pickle.load(open(next_matchday_path, "rb")))
            
            if len(next_matchday_df) == 0:
                print("next_matchday_df is empty")
                return None
            
            # Get the latest match start time (including time, not just date)
            latest_match_start = next_matchday_df["date"].max()
            
            # Get current matchday number and season
            current_matchday = next_matchday_df["matchDay"].iloc[0]
            current_season = next_matchday_df["season"].iloc[0]
            
            # Convert to datetime if needed
            if isinstance(latest_match_start, pd.Timestamp):
                latest_match_start = latest_match_start.to_pydatetime()
            elif not isinstance(latest_match_start, datetime.datetime):
                # If it's a date object, assume evening match at 20:30
                latest_match_start = datetime.datetime.combine(
                    latest_match_start, 
                    datetime.time(20, 30)
                )
            
            # Check if there is a next matchday (Edge Case: Spieltag 34 is the last)
            current_season_data = get_current_season(current_date)
            pickle_file = DATA_DIR / f"match_df_{current_season_data}.pck"
            
            has_next_matchday = False
            if pickle_file.exists():
                match_df = pd.DataFrame(pickle.load(open(pickle_file, "rb")))
                # Check if there are future matches after the current matchday
                future_matches = match_df[
                    (match_df["matchDay"] > current_matchday) & 
                    (match_df["season"] == current_season) &
                    (match_df["status"] == "future")
                ]
                has_next_matchday = len(future_matches) > 0
            
            if not has_next_matchday or current_matchday >= 34:
                # Last matchday of season - no next job to schedule
                print(f"Spieltag {current_matchday} is the last matchday of the season. No further job will be scheduled.")
                return None
            
            # Schedule for latest match start + 3 hours
            next_job_datetime = latest_match_start + datetime.timedelta(hours=3)
            
            print(f"Current matchday: {current_matchday}, latest match starts: {latest_match_start.strftime('%Y-%m-%d %H:%M')}, scheduling job for: {next_job_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # If the calculated time is in the past, schedule for 1 hour from now
        if next_job_datetime < current_date:
            next_job_datetime = current_date + datetime.timedelta(hours=1)
            print(f"Calculated time was in the past, scheduling for 1 hour from now: {next_job_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return next_job_datetime
    except Exception as e:
        print(f"Error scheduling next prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


class PredictionScheduler:
    """Scheduler for prediction jobs."""
    
    def __init__(self):
        self.scheduler = BlockingScheduler()
        self.next_job_id = None
    
    def job_wrapper(self):
        """Wrapper for scheduled job that sends email after execution."""
        print(f"\n{'='*60}")
        print(f"Scheduled job started at {datetime.datetime.now()}")
        print(f"{'='*60}\n")
        
        logs, next_job_datetime = run_prediction_job()
        
        # Add next job info to logs
        if next_job_datetime:
            logs += f"\n\nNächster geplanter Job: {next_job_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
            # Schedule the next regular job
            self.schedule_job(next_job_datetime)
        else:
            logs += "\n\n⚠️ KEIN WEITERER JOB EINGEPLANT"
            logs += "\nDies war vermutlich der letzte Spieltag der Saison (Spieltag 34)."
            logs += "\nDie Saison ist beendet. Keine weiteren Predictions werden automatisch geplant."
        
        # Send email
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        subject = f"[Football Prediction] Job Completed - {timestamp}"
        email_sent = send_email(subject, logs)
        if not email_sent:
            print("WARNING: Email could not be sent. Check SMTP configuration.")
    
    def schedule_job(self, job_datetime: datetime.datetime):
        """
        Schedule a prediction job at the specified datetime.
        
        Args:
            job_datetime: When to run the job
        """
        # Remove existing job if any
        if self.next_job_id:
            try:
                self.scheduler.remove_job(self.next_job_id)
            except:
                pass
        
        # Schedule new job
        job_id = f"prediction_job_{job_datetime.strftime('%Y%m%d_%H%M%S')}"
        self.scheduler.add_job(
            self.job_wrapper,
            trigger=DateTrigger(run_date=job_datetime),
            id=job_id,
        )
        self.next_job_id = job_id
        
        print(f"Scheduled prediction job for: {job_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_startup_test_job(self):
        """Run test job on startup."""
        print(f"\n{'='*60}")
        print("Running startup test job...")
        print(f"{'='*60}\n")
        
        logs, next_job_datetime = run_prediction_job()
        
        # Schedule the next regular job if available
        if next_job_datetime:
            logs += f"\n\nNächster geplanter Job: {next_job_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
            # Schedule the next regular job
            self.schedule_job(next_job_datetime)
        else:
            logs += "\n\n⚠️ KEIN WEITERER JOB EINGEPLANT"
            logs += "\nDies war vermutlich der letzte Spieltag der Saison (Spieltag 34)."
            logs += "\nDie Saison ist beendet. Keine weiteren Predictions werden automatisch geplant."
        
        # Send email
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        subject = f"[Football Prediction] Startup Test Job Completed - {timestamp}"
        email_sent = send_email(subject, logs)
        if not email_sent:
            print("WARNING: Email could not be sent. Check SMTP configuration.")
        
        print("\nStartup test job completed!")
    
    def start(self):
        """Start the scheduler."""
        print("Starting scheduler...")
        print(f"Current time: {datetime.datetime.now()}")
        
        # Run startup test job
        self.run_startup_test_job()
        
        # Start scheduler (will run continuously)
        print("\nScheduler running. Waiting for scheduled jobs...")
        try:
            self.scheduler.start()
        except KeyboardInterrupt:
            print("\nScheduler stopped by user")
        except Exception as e:
            print(f"Error in scheduler: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    scheduler = PredictionScheduler()
    scheduler.start()

