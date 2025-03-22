# run.bat
@echo off
echo Starting Spark and Flask application...

REM Start the Flask web application in the background
start "Flask App" python web_app.py

REM Keep the command prompt open (optional, for debugging)
echo Flask app started.  Press Ctrl+C to stop.
pause