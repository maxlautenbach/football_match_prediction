"""
Upload predictions to betting platform
Generates predictions directly from predict.py and uploads them to the betting platform
"""

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add BASE_DIR to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import prediction function (relative import since we're in scripts/)
from predict import generate_predictions


def upload_predictions_to_platform(results_df: pd.DataFrame, auto_submit: bool = False):
    """
    Upload predictions to betting platform using Selenium.
    
    Args:
        results_df: DataFrame with predictions
        auto_submit: Whether to automatically submit predictions
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
        from difflib import get_close_matches
        import dotenv
        
        # Load environment variables
        dotenv.load_dotenv()
        
        email = os.getenv("EMAIL")
        password = os.getenv("PASSWORT")
        link = os.getenv("LINK-TIPPABGABE")
        
        if not all([email, password, link]):
            print("Error: Missing environment variables (EMAIL, PASSWORT, LINK-TIPPABGABE)")
            print("Please create a .env file in the project root with these variables.")
            return
        
        print("Starting browser (headless mode)...")
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        import subprocess
        
        # Find Chrome and ChromeDriver binaries (try common locations)
        chrome_binary = None
        chromedriver_binary = None
        
        # Try common Chrome locations
        for path in ["/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/local/bin/google-chrome", "/opt/google/chrome/chrome"]:
            if os.path.exists(path) and os.access(path, os.X_OK):
                chrome_binary = path
                break
        
        # Try common ChromeDriver locations
        for path in ["/usr/bin/chromedriver", "/usr/local/bin/chromedriver"]:
            if os.path.exists(path) and os.access(path, os.X_OK):
                chromedriver_binary = path
                break
        
        if not chrome_binary:
            raise FileNotFoundError("Chrome binary not found in common locations")
        if not chromedriver_binary:
            raise FileNotFoundError("ChromeDriver not found in common locations")
        
        # Test Chrome binary
        try:
            result = subprocess.run(
                [chrome_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            print(f"Chrome version: {result.stdout.strip()}")
        except Exception as e:
            print(f"Warning: Could not verify Chrome version: {e}")
        
        # Test ChromeDriver binary
        try:
            result = subprocess.run(
                [chromedriver_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            print(f"ChromeDriver version: {result.stdout.strip()}")
        except Exception as e:
            print(f"Warning: Could not verify ChromeDriver version: {e}")
            print(f"This may indicate missing dependencies. Error: {e}")
        
        chrome_options = Options()
        # Essential headless and container options
        # Use old headless mode as it's more stable in containers
        chrome_options.add_argument("--headless")  # Use old headless mode (more stable)
        chrome_options.add_argument("--no-sandbox")  # Required for server environments
        chrome_options.add_argument("--disable-setuid-sandbox")  # Disable setuid sandbox
        chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
        chrome_options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration
        chrome_options.add_argument("--disable-extensions")  # Disable extensions
        chrome_options.add_argument("--disable-background-timer-throttling")  # For headless
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")  # For headless
        chrome_options.add_argument("--disable-renderer-backgrounding")  # For headless
        chrome_options.add_argument("--disable-features=TranslateUI,VizDisplayCompositor")  # Disable translation UI and compositor
        chrome_options.add_argument("--window-size=1920,1080")  # Set window size
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        chrome_options.add_argument("--remote-debugging-port=9222")  # Use fixed port for debugging
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # Hide automation
        chrome_options.add_argument("--disable-infobars")  # Disable infobars
        chrome_options.add_argument("--disable-logging")  # Disable logging
        chrome_options.add_argument("--log-level=3")  # Only fatal errors
        chrome_options.add_argument("--disable-default-apps")  # Disable default apps
        # Add experimental options for better container compatibility
        chrome_options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Use the installed Chrome binary instead of letting Selenium download a new one
        chrome_options.binary_location = chrome_binary
        
        # Use the installed ChromeDriver with additional service arguments
        service = Service(
            chromedriver_binary,
            service_args=['--verbose', '--log-path=/tmp/chromedriver.log']
        )
        
        print("Initializing Chrome WebDriver...")
        try:
            driver = webdriver.Chrome(service=service, options=chrome_options)
            print("Chrome WebDriver initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Chrome WebDriver: {e}")
            # Try to read ChromeDriver log if it exists
            try:
                if os.path.exists("/tmp/chromedriver.log"):
                    with open("/tmp/chromedriver.log", "r") as f:
                        log_content = f.read()
                        print(f"ChromeDriver log:\n{log_content}")
            except:
                pass
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
                    home_team = home_team_element.get_attribute("innerHTML").strip()
                    
                    if home_team and home_team != "":
                        # Find matching team in our predictions
                        matches = get_close_matches(home_team, results_df["Home_Team"].tolist(), n=1, cutoff=0.6)
                        
                        if matches:
                            selected_team = matches[0]
                            prediction_row = results_df[results_df["Home_Team"] == selected_team]
                            
                            if len(prediction_row) > 0:
                                prediction = prediction_row["Prediction"].iloc[0].split(":")
                                
                                # Fill in the prediction
                                inputs = datarow.find_elements(By.TAG_NAME, "input")
                                if len(inputs) >= 3:
                                    # Inputs[0] might be hidden, inputs[1] and inputs[2] are home/away goals
                                    inputs[1].clear()
                                    inputs[1].send_keys(prediction[0])
                                    inputs[2].clear()
                                    inputs[2].send_keys(prediction[1])
                                    
                                    print(f"  ✓ Filled: {selected_team} -> {prediction[0]}:{prediction[1]}")
                                    filled_count += 1
                                else:
                                    print(f"  ⚠ Warning: Could not find input fields for {selected_team}")
                        else:
                            print(f"  ⚠ Warning: No match found for '{home_team}'")
                except Exception as e:
                    print(f"  ⚠ Error processing row: {e}")
                    continue
            
            print(f"\nFilled {filled_count} predictions")
            
            if auto_submit:
                # Submit predictions
                try:
                    submit_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.NAME, "submitbutton"))
                    )
                    # Scroll to button
                    driver.execute_script("arguments[0].scrollIntoView(true);", submit_button)
                    # Wait a bit for any animations
                    import time
                    time.sleep(1)
                    submit_button.click()
                    print("Predictions submitted!")
                except ElementClickInterceptedException:
                    print("Warning: Submit button is blocked. Please submit manually.")
                except Exception as e:
                    print(f"Error submitting: {e}")
            else:
                print("\nPredictions filled but not submitted.")
                print("Review the predictions in the browser and submit manually if correct.")
                print("Or run with --submit flag to auto-submit.")
            
            # Keep browser open for review
            if not auto_submit:
                print("\nBrowser will stay open for review. Press Enter to close...")
                input()
        
        finally:
            driver.quit()
            print("Browser closed.")
        
    except ImportError:
        print("Error: Selenium or python-dotenv not installed.")
        print("Install with: uv pip install selenium python-dotenv")
    except Exception as e:
        print(f"Error during upload: {e}")
        import traceback
        traceback.print_exc()


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
        print(f"Error generating predictions: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

