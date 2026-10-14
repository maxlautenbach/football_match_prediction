"""
Upload predictions to betting platform
Generates predictions directly from predict.py and uploads them to the betting platform
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import shutil

_TAG_RE = re.compile(r"<[^>]+>")


def _plain_team_name(raw: str) -> str:
    """Strip HTML from Kicktipp col1 innerHTML."""
    return _TAG_RE.sub("", raw or "").replace("\xa0", " ").strip()

# Add repo root + scripts/ so both `models` and `predict` resolve
# (script execution already puts scripts/ on path; imports need both).
SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

# Import prediction function
from predict import generate_predictions


def _goal_inputs(datarow, By):
    """Return (home, away) tip input elements from a Kicktipp datarow, or None."""
    raw = datarow.find_elements(By.TAG_NAME, "input")
    inputs = [
        el
        for el in raw
        if (el.get_attribute("type") or "text").lower() in ("text", "number", "tel", "")
        and (el.get_attribute("name") or "").lower() != "submitbutton"
    ]
    # Kicktipp rows: hidden id + home + away (sometimes only the two score fields).
    visible = [el for el in inputs if el.is_displayed()]
    if len(visible) >= 2:
        return visible[0], visible[1]
    if len(inputs) >= 3:
        return inputs[1], inputs[2]
    if len(inputs) >= 2:
        return inputs[0], inputs[1]
    return None


def _read_tips_from_page(driver, By) -> dict[str, str]:
    """Map Kicktipp page home-team label → 'H:A' tip currently in the form."""
    table = driver.find_element(By.ID, "tippabgabeSpiele")
    tips: dict[str, str] = {}
    for datarow in table.find_elements(By.CLASS_NAME, "datarow"):
        try:
            home = _plain_team_name(
                datarow.find_element(By.CLASS_NAME, "col1").get_attribute("innerHTML") or ""
            )
            if not home:
                continue
            pair = _goal_inputs(datarow, By)
            if pair is None:
                continue
            home_el, away_el = pair
            h = (home_el.get_attribute("value") or "").strip()
            a = (away_el.get_attribute("value") or "").strip()
            if h != "" and a != "":
                tips[home] = f"{h}:{a}"
        except Exception:
            continue
    return tips


def _verify_tips_on_page(driver, By, results_df: pd.DataFrame, get_close_matches) -> tuple[int, list[str]]:
    """How many predicted tips are present on the tippabgabe form."""
    page_tips = _read_tips_from_page(driver, By)
    ok = 0
    missing: list[str] = []
    for _, row in results_df.iterrows():
        home = str(row["Home_Team"])
        want = str(row["Prediction"]).strip()
        matches = get_close_matches(home, list(page_tips.keys()), n=1, cutoff=0.6)
        if not matches:
            missing.append(f"{home} (not on page / empty)")
            continue
        got = page_tips[matches[0]]
        if got == want:
            ok += 1
        else:
            missing.append(f"{home}: want {want}, got {got}")
    return ok, missing


def upload_predictions_to_platform(results_df: pd.DataFrame, auto_submit: bool = False) -> int:
    """
    Upload predictions to betting platform using Selenium.
    
    Args:
        results_df: DataFrame with predictions
        auto_submit: Whether to automatically submit predictions

    Returns:
        Number of tips verified on Kicktipp after fill (and after submit if auto_submit).

    Raises:
        RuntimeError: if env/login/fill/submit verification fails.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
        from difflib import get_close_matches
        import dotenv
        import time
        
        # Load environment variables
        dotenv.load_dotenv()
        
        email = os.getenv("EMAIL")
        password = os.getenv("PASSWORT")
        link = os.getenv("LINK-TIPPABGABE")
        
        if not all([email, password, link]):
            raise RuntimeError(
                "Missing environment variables (EMAIL, PASSWORT, LINK-TIPPABGABE)"
            )
        
        print("Starting browser...")
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        import platform
        import subprocess

        # Find Chrome binary (macOS + common Linux paths)
        chrome_binary = None
        chrome_candidates = [
            "/opt/google/chrome/google-chrome",
            "/opt/google/chrome/chrome",
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/local/bin/google-chrome",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
        for path in chrome_candidates:
            if os.path.exists(path) and os.access(path, os.X_OK):
                chrome_binary = path
                break

        # Prefer an installed chromedriver; otherwise let Selenium Manager resolve it.
        chromedriver_binary = None
        for path in [
            "/usr/bin/chromedriver",
            "/opt/selenium/chromedriver",
            "/opt/homebrew/bin/chromedriver",
            "/usr/local/bin/chromedriver",
        ]:
            if os.path.exists(path) and os.access(path, os.X_OK):
                chromedriver_binary = path
                break

        # selenium/standalone-chrome often ships chromedriver on PATH
        if chromedriver_binary is None:
            which = shutil.which("chromedriver")
            if which:
                chromedriver_binary = which
        if chrome_binary is None:
            which = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
            if which:
                chrome_binary = which

        if not chrome_binary:
            raise FileNotFoundError(
                "Chrome binary not found. Install Google Chrome or set chrome_options.binary_location."
            )

        # Test Chrome binary
        try:
            result = subprocess.run(
                [chrome_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            print(f"Chrome version: {result.stdout.strip() or result.stderr.strip()}")
        except Exception as e:
            print(f"Warning: Could not verify Chrome version: {e}")

        if chromedriver_binary:
            try:
                result = subprocess.run(
                    [chromedriver_binary, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                print(f"ChromeDriver version: {result.stdout.strip()}")
            except Exception as e:
                print(f"Warning: Could not verify ChromeDriver version: {e}")
        else:
            print("No local chromedriver found — Selenium Manager will fetch a matching driver")

        chrome_options = Options()
        # Headless is fine for auto-submit; keep headed on macOS for manual review.
        if auto_submit or platform.system() != "Darwin":
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-setuid-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option(
            "excludeSwitches", ["enable-logging", "enable-automation"]
        )
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.binary_location = chrome_binary

        print("Initializing Chrome WebDriver...")
        try:
            if chromedriver_binary:
                service = Service(chromedriver_binary)
                driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                driver = webdriver.Chrome(options=chrome_options)
            print("Chrome WebDriver initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Chrome WebDriver: {e}")
            raise
        
        try:
            # Navigate to login page
            print(f"Navigating to: {link}")
            driver.get(link)
            
            # Login process
            print("Logging in...")
            try:
                nav_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "navtoggle"))
                )
                nav_button.click()
            except TimeoutException:
                print("Warning: Navigation button not found, trying to proceed...")
            
            # Enter credentials
            email_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "kennung"))
            )
            email_input.clear()
            email_input.send_keys(email)
            
            passwort_input = driver.find_element(By.ID, "passwort")
            passwort_input.clear()
            passwort_input.send_keys(password)
            
            login_button = driver.find_element(By.NAME, "submitbutton")
            login_button.click()
            
            # Wait for login to complete
            print("Waiting for login to complete...")
            WebDriverWait(driver, 15).until(
                lambda d: "login" not in d.current_url.lower() or d.find_elements(By.ID, "tippabgabeSpiele")
            )
            
            import time
            time.sleep(2)  # Wait for page to fully load after login
            
            # Handle cookie banner if present (after login)
            print("Checking for cookie banner...")
            
            try:
                # Try to find and close cookie banner
                cookie_selectors = [
                    "//iframe[contains(@src, 'privacy')]",
                    "//iframe[contains(@id, 'sp_message')]",
                    "//iframe[contains(@title, 'SP Consent')]",
                    "//div[contains(@class, 'cookie')]//button",
                ]
                
                cookie_handled = False
                for selector in cookie_selectors:
                    try:
                        if "iframe" in selector:
                            # Try to find iframe
                            frames = driver.find_elements(By.XPATH, selector)
                            if frames:
                                frame = frames[0]
                                driver.switch_to.frame(frame)
                                time.sleep(1)  # Wait for iframe content to load
                                
                                # Try multiple button selectors
                                button_selectors = [
                                    "//button[contains(text(), 'Accept')]",
                                    "//button[contains(text(), 'Akzeptieren')]",
                                    "//button[contains(@id, 'accept')]",
                                    "//button[contains(@class, 'accept')]",
                                    "//button[contains(@aria-label, 'Accept')]",
                                ]
                                
                                for btn_selector in button_selectors:
                                    try:
                                        accept_button = WebDriverWait(driver, 3).until(
                                            EC.element_to_be_clickable((By.XPATH, btn_selector))
                                        )
                                        accept_button.click()
                                        time.sleep(1)  # Wait for click to register
                                        cookie_handled = True
                                        print("Cookie banner handled (iframe)")
                                        break
                                    except:
                                        continue
                                
                                driver.switch_to.default_content()
                                if cookie_handled:
                                    break
                        else:
                            # Try direct button click
                            buttons = driver.find_elements(By.XPATH, selector)
                            if buttons:
                                buttons[0].click()
                                time.sleep(1)  # Wait for click to register
                                print("Cookie banner handled (direct)")
                                cookie_handled = True
                                break
                    except Exception as e:
                        continue
                
                if not cookie_handled:
                    print("No cookie banner found or already handled")
                else:
                    time.sleep(1)  # Additional wait after handling cookie banner
                        
            except Exception as e:
                print(f"Cookie banner handling error: {e}")
                driver.switch_to.default_content()
            
            # Navigate to prediction page
            print("Navigating to prediction page...")
            tip_md = int(results_df["Matchday"].iloc[0]) if "Matchday" in results_df.columns else "?"
            tip_season = int(results_df["Season"].iloc[0]) if "Season" in results_df.columns else "?"
            print(f"Uploading tips for season {tip_season} matchday {tip_md} ({len(results_df)} matches)")
            driver.get(link)
            time.sleep(2)  # Wait for page to load
            
            # Wait for prediction table
            print("Waiting for prediction table...")
            tippabgabe_tabelle = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "tippabgabeSpiele"))
            )
            time.sleep(1)  # Additional wait for table to be fully loaded
            
            # Fill in predictions
            print("Filling in predictions...")
            filled_count = 0
            
            for datarow in tippabgabe_tabelle.find_elements(By.CLASS_NAME, "datarow"):
                try:
                    home_team_element = datarow.find_element(By.CLASS_NAME, "col1")
                    home_team = _plain_team_name(
                        home_team_element.get_attribute("innerHTML") or ""
                    )
                    
                    if home_team and home_team != "":
                        # Find matching team in our predictions
                        matches = get_close_matches(home_team, results_df["Home_Team"].tolist(), n=1, cutoff=0.6)
                        
                        if matches:
                            selected_team = matches[0]
                            prediction_row = results_df[results_df["Home_Team"] == selected_team]
                            
                            if len(prediction_row) > 0:
                                prediction = prediction_row["Prediction"].iloc[0].split(":")
                                pair = _goal_inputs(datarow, By)
                                if pair is not None:
                                    home_input, away_input = pair
                                    home_input.clear()
                                    home_input.send_keys(prediction[0])
                                    away_input.clear()
                                    away_input.send_keys(prediction[1])
                                    
                                    print(
                                        f"  ✓ Filled: page='{home_team}' → {selected_team} "
                                        f"{prediction[0]}:{prediction[1]}"
                                    )
                                    filled_count += 1
                                else:
                                    print(f"  ⚠ Warning: Could not find input fields for {selected_team}")
                        else:
                            print(f"  ⚠ Warning: No match found for '{home_team}'")
                except Exception as e:
                    print(f"  ⚠ Error processing row: {e}")
                    continue
            
            print(f"\nFilled {filled_count} predictions")
            expected = len(results_df)
            if filled_count == 0:
                raise RuntimeError("Filled 0 predictions — Kicktipp page/team matching failed")
            if filled_count < expected:
                print(
                    f"WARNING: only filled {filled_count}/{expected} tips "
                    "(page may show a subset or names did not match)"
                )

            pre_ok, pre_missing = _verify_tips_on_page(
                driver, By, results_df, get_close_matches
            )
            print(f"Pre-submit form check: {pre_ok}/{expected} tips present in inputs")
            if pre_ok == 0:
                raise RuntimeError(
                    "Form inputs empty after fill — values did not stick: "
                    + "; ".join(pre_missing[:5])
                )
            
            if auto_submit:
                submit_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.NAME, "submitbutton"))
                )
                driver.execute_script("arguments[0].scrollIntoView(true);", submit_button)
                time.sleep(1)
                try:
                    submit_button.click()
                except ElementClickInterceptedException as e:
                    raise RuntimeError(
                        "Submit button blocked (cookie/overlay?) — tips not saved"
                    ) from e

                # Let Kicktipp finish the POST before we tear down Chrome.
                time.sleep(3)
                try:
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.ID, "tippabgabeSpiele"))
                    )
                except TimeoutException:
                    # Some responses navigate away briefly; reload tippabgabe.
                    driver.get(link)
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.ID, "tippabgabeSpiele"))
                    )
                time.sleep(1)

                # Hard check: reload tippabgabe and confirm tips persisted.
                driver.get(link)
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.ID, "tippabgabeSpiele"))
                )
                time.sleep(1)
                post_ok, post_missing = _verify_tips_on_page(
                    driver, By, results_df, get_close_matches
                )
                print(f"Post-submit verify: {post_ok}/{expected} tips persisted on Kicktipp")
                if post_ok < max(1, expected // 2):
                    detail = "; ".join(post_missing[:8]) or "no tip fields readable"
                    raise RuntimeError(
                        f"Submit click ran but Kicktipp did not keep tips "
                        f"({post_ok}/{expected}): {detail}"
                    )
                print("Predictions submitted and verified!")
                return post_ok
            else:
                print("\nPredictions filled but not submitted.")
                print("Review the predictions in the browser and submit manually if correct.")
                print("Or run with --submit flag to auto-submit.")
                print("\nBrowser will stay open for review. Press Enter to close...")
                input()
                return filled_count
        
        finally:
            driver.quit()
            print("Browser closed.")
        
    except ImportError as e:
        raise RuntimeError(
            "Selenium or python-dotenv not installed. "
            "Install with: uv pip install selenium python-dotenv"
        ) from e
    except Exception:
        # Ensure scheduler/CLI see real failures (do not swallow).
        raise


def main():
    """
    Main function to upload predictions.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload predictions to betting platform")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Automatically submit predictions after filling",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Upload Predictions to Betting Platform")
    print("=" * 60)
    
    # Generate predictions directly from predict.py
    print("\nGenerating predictions...")
    try:
        results_df = generate_predictions(save_csv=False, verbose=False)
        print(f"Generated {len(results_df)} predictions")
        
        # Display predictions
        print("\nPredictions to upload:")
        for idx, row in results_df.iterrows():
            print(f"  {row['Home_Team']} vs {row['Away_Team']}: {row['Prediction']}")
        
        # Confirm
        if not args.submit:
            print("\n" + "=" * 60)
            response = input("Upload these predictions? (y/n): ")
            if response.lower() != 'y':
                print("Upload cancelled.")
                return
        
        # Upload
        print("\n" + "=" * 60)
        upload_predictions_to_platform(results_df, auto_submit=args.submit)
        
    except Exception as e:
        print(f"Error during predict/upload: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()

