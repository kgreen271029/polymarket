#!/usr/bin/env python3
"""
get_robinhood_token.py — Run this LOCALLY on your machine to get your
Robinhood Agentic Trading Bearer token for the trading bot.

Usage:
    python get_robinhood_token.py

Requirements: Python 3.8+ (no extra packages needed — stdlib only)
"""

import base64
import hashlib
import json
import secrets
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

REGISTER_URL = "https://agent.robinhood.com/oauth/trading/register"
AUTH_URL     = "https://robinhood.com/oauth"
TOKEN_URL    = "https://api.robinhood.com/oauth2/token/"
REDIRECT_URI = "http://localhost:8765/callback"
SCOPE        = "internal"
PORT         = 8765


def _pkce() -> tuple[str, str]:
    """Return (verifier, challenge) PKCE pair."""
    verifier  = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


def _register_client() -> dict:
    """Dynamic client registration — creates a fresh OAuth client."""
    payload = json.dumps({
        "redirect_uris":              [REDIRECT_URI],
        "client_name":                "Agentic Trading Bot",
        "token_endpoint_auth_method": "none",
        "grant_types":                ["authorization_code", "refresh_token"],
        "response_types":             ["code"],
        "scope":                      SCOPE,
    }).encode()
    req  = urllib.request.Request(
        REGISTER_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read())


def _exchange_code(client_id: str, code: str, verifier: str) -> dict:
    """Exchange authorization code + PKCE verifier for tokens."""
    payload = urllib.parse.urlencode({
        "grant_type":    "authorization_code",
        "client_id":     client_id,
        "code":          code,
        "redirect_uri":  REDIRECT_URI,
        "code_verifier": verifier,
    }).encode()
    req  = urllib.request.Request(
        TOKEN_URL, data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read())


_auth_code: list[str | None] = [None]


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if "code" in params:
            _auth_code[0] = params["code"][0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<h2 style='font-family:sans-serif;color:green'>"
            b"Authorized! You can close this tab and return to the terminal.</h2>"
        )

    def log_message(self, *_):
        pass


def main() -> None:
    print("\n=== Robinhood Agentic Trading — Token Helper ===\n")

    print("Step 1/3: Registering OAuth client with Robinhood...")
    try:
        client_info = _register_client()
    except Exception as e:
        print(f"ERROR during client registration: {e}")
        print("Check your internet connection and try again.")
        return
    client_id = client_info["client_id"]
    print(f"  Client ID: {client_id}\n")

    verifier, challenge = _pkce()
    state = secrets.token_hex(16)

    auth_params = urllib.parse.urlencode({
        "client_id":             client_id,
        "redirect_uri":          REDIRECT_URI,
        "response_type":         "code",
        "scope":                 SCOPE,
        "code_challenge":        challenge,
        "code_challenge_method": "S256",
        "state":                 state,
    })
    auth_url = f"{AUTH_URL}?{auth_params}"

    # Start local callback server in background
    server = HTTPServer(("localhost", PORT), _CallbackHandler)
    t = Thread(target=server.handle_request, daemon=True)
    t.start()

    print("Step 2/3: Opening Robinhood authorization page in your browser...")
    print(f"  If it doesn't open automatically, go to:\n  {auth_url}\n")
    webbrowser.open(auth_url)

    print("  Waiting for you to authorize in Robinhood (up to 2 min)...")
    t.join(timeout=120)

    if not _auth_code[0]:
        print("\nERROR: Timed out — no callback received.")
        print("Make sure nothing is blocking port 8765 and try again.")
        return

    print("\nStep 3/3: Exchanging authorization code for Bearer token...")
    try:
        token_data = _exchange_code(client_id, _auth_code[0], verifier)
    except Exception as e:
        print(f"ERROR during token exchange: {e}")
        return

    access_token  = token_data.get("access_token", "")
    refresh_token = token_data.get("refresh_token", "")
    expires_in    = token_data.get("expires_in", "?")

    if not access_token:
        print("ERROR: Token exchange succeeded but no access_token in response.")
        print("Full response:", token_data)
        return

    print("\n" + "=" * 62)
    print("SUCCESS — add these lines to your .env file:")
    print("=" * 62)
    print(f"ROBINHOOD_MCP_TOKEN={access_token}")
    if refresh_token:
        print(f"ROBINHOOD_REFRESH_TOKEN={refresh_token}")
    print(f"ROBINHOOD_CLIENT_ID={client_id}")
    print("=" * 62)
    print(f"\nToken expires in: {expires_in} seconds")
    print("The bot will auto-refresh using ROBINHOOD_REFRESH_TOKEN.")
    print("Paste all three values into your .env, then run: python main.py\n")


if __name__ == "__main__":
    main()
