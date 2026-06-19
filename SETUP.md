# 🤖 FREE Trading Bot - Phone Setup (2 minutes!)

## What You Need (ONLY 1 thing for free trading!)

### ✅ GROQ API KEY (FREE - Required)
This is a FREE AI service. Takes 1 minute on your phone:

1. **Open this link on your phone:**
   ```
   https://console.groq.com/keys
   ```

2. **Sign up** with your email (takes 30 seconds)

3. **Copy your API key** (looks like: `gsk_xxxxxxxxxxxxx`)

4. **Send it to Claude** and ask to set it in your .env file, OR:
   - Open a terminal and run:
   ```bash
   echo "GROQ_API_KEY=gsk_your_key_here" >> .env
   ```
   (Replace `gsk_your_key_here` with your actual key)

That's it! 🎉

---

## 🚀 Bot Now Running FREE!

The bot will:
- ✅ Analyze stocks every 5 minutes (using FREE Groq AI)
- ✅ Generate BUY/SELL signals 
- ✅ Run in dry-run mode (simulated trades)
- ✅ Keep running all day (watchdog)

---

## Optional: Real Trading Setup

### Robinhood Trading (Add Later)
To actually execute trades, you need Robinhood credentials:

1. **In Robinhood app on your phone:**
   - Settings → Security → API Settings
   - Generate MCP Token
   - Copy the token

2. **Tell Claude:** "Set my Robinhood MCP token to: [your-token]"

3. **Telegram Alerts (Optional):**
   - Create a Telegram bot with BotFather
   - Get your Chat ID
   - Tell Claude: "Set my Telegram token"

---

## 📊 Monitor Your Bot

Watch it trade in real-time:
```bash
bash monitor.sh
```

---

## ✨ Features Unlocked

| Feature | Free | Paid |
|---------|------|------|
| AI Analysis | ✅ Groq (FREE) | Anthropic |
| Dry Run | ✅ Yes | Yes |
| Real Trading | ❌ (need Robinhood) | ✅ Yes |
| Alerts | ❌ (need Telegram) | ✅ Yes |

---

## Need Help?

Already running in **DRY-RUN MODE** analyzing stocks with FREE Groq AI! 🚀

Just add your Groq key and it's fully functional!
