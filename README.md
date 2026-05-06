# Book Recommendation System

## Run locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   python app.py
   ```
3. Open `http://127.0.0.1:5000/`

## Deploy to a Python host

This repository contains a Flask app, so it cannot be served directly through GitHub Pages.

Recommended services:
- Render
- Railway
- Heroku

A `Procfile` is included for deployment to services that support Gunicorn.
