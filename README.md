## Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create env and add API keys
cp env_template.txt .env

# Run voice agent
python3 main.py
```

Voice agent will be available via http://localhost:7860/client/

Built with [Pipecat](https://github.com/pipecat-ai/pipecat)

## Live Demo

https://latex-registered-moreover-mens.trycloudflare.com/client/

Enable microphone, Click "Connect", and Gaia will greet you.
