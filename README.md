python main.py

ps aux | grep [l]ightning | grep -v grep | awk '{print $2}' | xargs kill -9