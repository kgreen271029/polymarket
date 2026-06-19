#!/usr/bin/env python3
"""
Simple Robinhood credential setup for phone users
Walks you through getting credentials and enables real trading
"""

import os
import sys

def main():
    print("\n" + "="*70)
    print("🤖 ROBINHOOD REAL TRADING SETUP")
    print("="*70)
    print("\nThis will enable REAL trading on your phone.\n")

    print("STEP 1: Get your Robinhood MCP Token")
    print("-" * 70)
    print("""
On your phone, open Robinhood app:
1. Tap Account (bottom right)
2. Tap Settings
3. Look for "Security" or "Developer" or "API"
4. Find or Generate: MCP_TOKEN
5. Copy the long string that starts with something like:
   - "eyJ..." or
   - "mcp_..." or
   - some long token

Then paste it here:
""")

    mcp_token = input("🔐 Paste MCP_TOKEN here: ").strip()
    if not mcp_token:
        print("❌ No token provided")
        return False

    print("\n✅ Got MCP Token")

    print("\nSTEP 2: Get your REFRESH_TOKEN")
    print("-" * 70)
    print("""
Still in Robinhood settings, look for:
- "Refresh Token" or
- "Session Token" or
- "Auth Token"

Paste it:
""")

    refresh_token = input("🔐 Paste REFRESH_TOKEN here: ").strip()
    if not refresh_token:
        print("❌ No token provided")
        return False

    print("\n✅ Got Refresh Token")

    print("\nSTEP 3: Get your CLIENT_ID")
    print("-" * 70)
    print("""
In the same settings area, look for:
- "Client ID" or
- "App ID" or
- "Account ID"

Paste it:
""")

    client_id = input("🔐 Paste CLIENT_ID here: ").strip()
    if not client_id:
        print("❌ No token provided")
        return False

    print("\n✅ Got Client ID")

    # Write to .env
    print("\n" + "="*70)
    print("💾 SAVING CREDENTIALS")
    print("="*70)

    env_file = ".env"

    # Read existing .env
    env_content = ""
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            env_content = f.read()

    # Remove old Robinhood lines
    lines = [line for line in env_content.split('\n')
             if not line.startswith('ROBINHOOD_')]

    # Add new credentials
    env_content = '\n'.join(lines).strip() + '\n'
    env_content += f"ROBINHOOD_MCP_TOKEN={mcp_token}\n"
    env_content += f"ROBINHOOD_REFRESH_TOKEN={refresh_token}\n"
    env_content += f"ROBINHOOD_CLIENT_ID={client_id}\n"

    # Write back
    with open(env_file, 'w') as f:
        f.write(env_content)

    print("✅ Credentials saved to .env")
    print("\n" + "="*70)
    print("🎉 SETUP COMPLETE!")
    print("="*70)
    print("""
Your bot will now execute REAL trades on Robinhood!

The bot will:
✅ Analyze stocks every 5 minutes
✅ Execute real buy/sell orders
✅ Track actual positions and P&L
✅ All based on momentum strategy

Trades start immediately when market is open!

Check logs for trade execution:
  tail -f trading_bot.log

View your trading account:
  cat paper_trading.json

""")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
