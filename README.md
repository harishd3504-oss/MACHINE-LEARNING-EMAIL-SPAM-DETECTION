# Email Spam Detection (SMS Spam) - Simple Example

This project shows a minimal SMS spam detection pipeline using Python and scikit-learn.

Quick start

1. (Optional) Place the full dataset in the project folder as `spam.csv` or `SMSSpamCollection`.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the script to train and evaluate using the included small sample dataset:

```powershell
python spam_detector.py
```

4. Predict a custom message after training:

```powershell
python spam_detector.py --predict "You've won a prize, reply now"
```

Notes
- The script will prefer `spam.csv` or `SMSSpamCollection` if present; otherwise it uses `sample_sms.csv`.
- `spam.csv` from some sources may contain extra columns; the loader attempts to handle common formats.
