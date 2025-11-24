# Liquid-Granite 

## (Granite-4.0-350m Generator + LiquidAI-LFM2-350m Critic)

<details>
  <summary>Metadata</summary>

  <ul>
    <li><strong>Date:</strong> 2025-11-24</li>
    <li><strong>Version:</strong> 1.0</li>
    <li><strong>Author:</strong> Julian A. Gonzalez</li>
    <li><strong>License:</strong> MIT</li>
  </ul>

</details>

---

### Overview

A lightweight, production-ready **Researcher–Critic** pattern that runs **entirely locally** via Ollama.  

**Models**:

Generator uses:
- `hf.co/unsloth/granite-4.0-350m-GGUF:Q4_K_M`

Critic uses:
- `hf.co/LiquidAI/LFM2-350M-GGUF`

---

## ✨ Features
- **Zero cloud calls** – full privacy
- **Streaming-ready** (CLI flag `--stream`)
- **Automatic retries** with exponential back-off
- **Early convergence** stop (regex based)
- **Beautiful terminal UI** (Rich)
- **Extensible** – swap LLM back-ends by implementing `LLMClient` protocol

---

## 🚀 Quick Start

1. Install **Ollama**  
   macOS/Linux: `curl -fsSL https://ollama.ai/install.sh | sh`  
   Windows: download from [ollama.ai](https://ollama.ai)

2. Pull the models
   ```bash
   ollama pull hf.co/unsloth/granite-4.0-350m-GGUF:Q4_K_M
   ollama pull hf.co/LiquidAI/LFM2-350M-GGUF:Q4_K_M
   ```

3. Clone / copy this repo
   
```bash
   git clone <repo> && cd <repo>
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. Run
   
```bash
   python liquidpi.py
   # or with streaming
   python liquipi.py --stream
   ```

---

📖 CLI Usage

```
usage: liquidpi.py [-h] [--stream] [--iterations N] [--quiet]

optional arguments:
  -h, --help       show this help message and exit
  --stream         Enable token streaming (lower latency)
  --iterations N   Max refinement loops (default 3)
  --quiet          Suppress Rich UI (plain text)
```

---

🧠 How It Works
1. Generator produces an answer.  
2. Critic scores it and gives concrete feedback.  
3. Loop repeats until “no significant improvements needed” or max iterations.  
4. Final answer is printed + saved to `history` list.

---

## 🔧 Extending / Self-Hosting
- Implement `LLMClient` protocol to plug OpenAI, Gemini, etc.  
- Change `GENERATOR_MODEL` / `CRITIC_MODEL` constants for new models.  
- Wrap with FastAPI / Flask – the core `DualAgentCoordinator` is I/O agnostic.

---

### 📄 License

Apache 2.0 – see `LICENSE` file.

---


## requirements.txt

```

Core
requests>=2.31.0
rich>=13.7.0

# Optional dev / quality tools (uncomment if desired)

# black>=23.0
# isort>=5.12
# mypy>=1.5
# flake8>=6.0

```

**@2025** [creativeact.net](www.creativeact.net)
