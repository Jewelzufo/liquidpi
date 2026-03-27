# Liquid-Granite

## (Granite-4.0-350m Generator + LiquidAI-LFM2-350m Critic)

<details>
  <summary>Metadata</summary>

  <ul>
    <li><strong>Date:</strong> 2025-11-24</li>
    <li><strong>Version:</strong> 1.0</li>
    <li><strong>Author:</strong> Julian A. Gonzalez</li>
    <li><strong>License:</strong> Apache 2.0</li>
  </ul>

</details>

---

### Overview

A lightweight, **Researcher–Critic** pattern that runs **entirely locally** via Ollama.

>**Tested** | **Device**: *Raspberry Pi 5 (8gb RAM, MicroSD)* | **Created**: *11/2025*, **Updated:** 03/27/2026 |

**Models**:

**Generator uses:**

- `lfm2.5-thinking`

**Critic uses:**

- `granite4:350m-h`

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
   ollama pull granite4:350m-h
   ollama pull lfm2.5-thinking
3. Clone / copy this repo

```bash
   git clone https://github.com/Jewelzufo/liquid-granite && cd
   liquid-granite
```

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

1. Run

```bash
   python liquidpi.py
   # or with streaming
   python liquipi.py --stream
   ```

---

<br>

# 📖 CLI Usage

```
usage: liquidpi.py [-h] [--stream] [--iterations N] [--quiet]

optional arguments:
  -h, --help       show this help message and exit
  --stream         Enable token streaming (lower latency)
  --iterations N   Max refinement loops (default 3)
  --quiet          Suppress Rich UI (plain text)
```

---

<br>

# 🧠 How It Works

1. Generator produces an answer.  
2. Critic scores it and gives concrete feedback.  
3. Loop repeats until “no significant improvements needed” or max iterations.  
4. Final answer is printed + saved to `history` list.

---

<br>

# 🔧 Extending / Self-Hosting

- Implement `LLMClient` protocol to plug OpenAI, Gemini, etc.  
- Change `GENERATOR_MODEL` / `CRITIC_MODEL` constants for new models.  
- Wrap with FastAPI / Flask – the core `DualAgentCoordinator` is I/O agnostic.

---

# 📄 License

**Apache 2.0** – see `LICENSE` file.

<br>

@2026 **CreativeAct Technologies**
