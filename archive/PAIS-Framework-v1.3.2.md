# PRIVATE AI SOVEREIGNTY (PAIS)

**A Framework for Consumer Control of Local AI Infrastructure**

**Version 1.3.2**  
**February 2026**

*Includes: Storage Architecture, Tailscale Networking, Containerization Strategy, Observability, Security Hardening, Local-First with Cloud Overflow, Multi-Tenancy, OpenClaw Integration, and Flexible Orchestration*

---

> *"If I build the tools, I have only myself to blame for the exposure and consequences of my and my family's use."*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What's New in Version 1.3.2](#2-whats-new-in-version-132)
3. [Hardware Architecture](#3-hardware-architecture)
4. [Orchestration Layer](#4-orchestration-layer)
5. [Inference Architecture](#5-inference-architecture)
6. [OpenClaw Personal AI Assistant](#6-openclaw-personal-ai-assistant)
7. [Cloud Overflow Strategy](#7-cloud-overflow-strategy)
8. [Containerization Strategy](#8-containerization-strategy)
9. [Source Code Management](#9-source-code-management)
10. [Observability Stack](#10-observability-stack)
11. [Security and Hardening](#11-security-and-hardening)
12. [Knowledge Layer](#12-knowledge-layer)
13. [Multi-Tenancy and Governance](#13-multi-tenancy-and-governance)
14. [Performance Optimization](#14-performance-optimization)
15. [Backup and Disaster Recovery](#15-backup-and-disaster-recovery)
16. [Kubernetes Migration Roadmap](#16-kubernetes-migration-roadmap)
17. [Fine-Tuning Infrastructure](#17-fine-tuning-infrastructure)
18. [Multi-Modal Capabilities](#18-multi-modal-capabilities)
19. [Quick Reference](#19-quick-reference)

---

## 1. Executive Summary

### 1.1 The Problem

Artificial intelligence has become essential infrastructure for knowledge work, creativity, and daily life. Yet the dominant delivery model—cloud-based APIs—creates fundamental tensions:

- **Loss of Control**: Terms of service change unilaterally; pricing increases without notice; capabilities are deprecated or restricted
- **Privacy Erosion**: Every prompt, document, and query transits third-party infrastructure with opaque data handling
- **Rising Costs**: Token-based pricing creates unpredictable expenses that scale with usage
- **Dependency Risk**: Critical workflows become hostage to provider availability, policy changes, and geopolitical factors

### 1.2 The PAIS Solution

Private AI Sovereignty (PAIS) is a framework for individuals and families to establish **local-first AI infrastructure with intelligent cloud overflow**. By leveraging modern local hardware, open-source models, containerized services, secure mesh networking, and flexible orchestration tools, PAIS enables:

- **Local-first inference**—Ollama and local models handle 95%+ of workloads on-device
- **Intelligent cloud overflow**—Seamlessly route to NVIDIA Brev, Google Vertex AI, Gemini, or specialized services (ElevenLabs) when needed
- **Full data sovereignty**—Explicit control over when data leaves your network
- **Proactive AI assistance**—OpenClaw provides 24/7 personal AI assistant with system integration
- **Customizable governance**—You define content policies, access controls, and cloud usage thresholds
- **Secure remote access**—Tailscale mesh VPN provides seamless connectivity without exposing services to the internet
- **Portable architecture**—Docker containers enable workload mobility across any PAIS node
- **Observable systems**—Prometheus and Grafana provide comprehensive metrics and performance insights
- **Production-grade security**—Container hardening, zero-trust networking, and encrypted backups
- **Flexible orchestration**—Choose from LangFlow, LMStudio, AnythingLLM, n8n, OpenClaw, or combined approaches

---

## 2. What's New in Version 1.3.2

Version 1.3.2 incorporates updated hardware specifications and adds OpenClaw personal AI assistant integration.

### 2.1 Hardware Updates

- **Mac Studio M3 Ultra**: Updated to 256GB unified memory (from 128GB/192GB)
- **GMKtec K6**: Updated to 64GB DDR5 RAM (from 32GB)
- **New mobile workstations**: ThinkPad P14s Gen 6 AMD (96GB), MacBook Pro M4 Pro (48GB), Dell Pro Max 16 Plus (32GB)

### 2.2 OpenClaw Integration

- **OpenClaw**: Open-source personal AI assistant with 24/7 proactive capabilities
- **Deployment**: Dedicated GMKtec K6 (64GB RAM) for always-on operation
- **Capabilities**: Email management, calendar scheduling, file organization, system monitoring, browser automation
- **Local-first**: Runs on Ollama with optional cloud fallback

### 2.3 Expanded Device Roles

- **GMKtec K6**: Now OpenClaw host + secondary orchestration/development
- **Mobile workstations**: Three devices for on-the-go AI capabilities
- **Mac Studio**: Increased capacity supports 236B models

---

## 3. Hardware Architecture

### 3.1 Reference Hardware Stack

The PAIS reference architecture is designed to scale from single-node setups to distributed deployments across multiple device types.

#### Orchestration Layer

```yaml
Mac mini M4 Pro:
  cpu: 14-core (10P + 4E)
  unified_memory: 48GB
  neural_engine: 16-core
  storage: 1TB SSD
  power: ~50W
  qty: 1
  role: "Primary control plane, orchestration tools, web services"
  
  services:
    - LangFlow (visual workflows)
    - LMStudio (local model testing)
    - AnythingLLM (RAG/chat interface)
    - n8n (automation)
    - Prometheus & Grafana (observability)
    - API Gateway (Traefik)
    - Model Router (intelligent tiering + cloud overflow)
```

#### Inference Layer

```yaml
Mac Studio M3 Ultra:
  unified_memory: 256GB
  cpu: 32-core (24P + 8E)
  gpu: 80-core
  neural_engine: 32-core
  storage: 4TB SSD
  power: ~150W
  qty: 1
  role: "Primary local inference via Ollama, ComfyUI, multi-modal"
  
  inference_engine: Ollama
  supported_models:
    - Llama 3.2 70B (Q4_K_M): 6-7 tok/s
    - Qwen 2.5 72B (Q4_K_M): 5-6 tok/s
    - DeepSeek v3 236B (Q4_K_M): 2-3 tok/s (new with 256GB)
    - Mistral Small 22B: 12-15 tok/s
    - Stable Diffusion XL: 25-35s per image
    - Whisper Medium: real-time transcription

DGX Spark #1:
  gb10_grace_blackwell: "20-core Arm CPU + 2× tensor units"
  memory: 128GB LPDDR5x
  storage: 4TB NVMe
  qty: 2 units
  role: "Batch inference, high-concurrency Ollama server"
  
  inference_engine: Ollama (or vLLM for advanced use cases)
  use_case_1: "70B model inference (multi-turn, batch processing)"
  use_case_2: "180B+ model experiments (rare, overflow to cloud preferred)"

DGX Spark #2:
  gb10_grace_blackwell: "20-core Arm CPU + 2× tensor units"
  memory: 128GB LPDDR5x
  storage: 4TB NVMe
  qty: 1 unit
  role: "Fine-tuning, LoRA training, model experimentation"

GMKtec AI Max+ 395:
  cpu: Ryzen AI 9 395 (16-core, 5.1 GHz boost)
  unified_memory: 128GB LPDDR5X
  npu: 40+ TOPS AI accelerator
  storage: 2TB PCIe 4.0 SSD
  power: ~65W
  qty: 1
  role: "Knowledge layer, embeddings, document processing"
  
  primary_services:
    - Qdrant vector database (64GB allocation)
    - Redis semantic cache (32GB allocation)
    - Embedding generation (nomic-embed on NPU)
    - Document ingestion and chunking
```

#### OpenClaw & Development Layer

```yaml
GMKtec K6:
  cpu: AMD Ryzen 7 7840HS (8-core, up to 5.1 GHz)
  memory: 64GB DDR5 (16GB × 4)
  storage: 1TB PCIe 4.0 SSD
  networking: Dual NIC 2.5Gbps, WiFi 6E, USB4
  power: ~45W
  qty: 1
  role: "OpenClaw AI assistant + secondary orchestration + development"
  
  primary_services:
    - OpenClaw (personal AI assistant, 24/7)
    - Ollama (Mistral Small 22B or Qwen 2.5 14B)
    - Secondary n8n instance (development/backup)
    - CI/CD runner (GitHub Actions self-hosted)
    - Model evaluation and A/B testing
  
  rationale:
    - 64GB RAM sufficient for OpenClaw + 22B model
    - Dual 2.5Gbps NIC for network redundancy
    - Always-on, separate from main orchestration
    - USB4 for fast NAS access
```

#### Mobile Workstations

```yaml
ThinkPad P14s Gen 6 AMD:
  cpu: AMD Ryzen AI 9 HX PRO 370 (up to 5.1 GHz)
  memory: 96GB DDR5-5600MT/s (2 × 48GB)
  gpu: Integrated AMD Radeon 890M
  npu: AMD Ryzen AI (40+ TOPS)
  storage: 2TB SSD M.2 PCIe Gen4
  display: 14" WUXGA IPS, 100% sRGB, 500 nits
  power: 57Wh battery
  qty: 1
  role: "Mobile AI workstation, on-the-go inference, development"
  
  capabilities:
    - Ollama (13B-22B models on CPU, 7B on NPU)
    - Local development environment
    - Model testing and evaluation
    - RAG with AnythingLLM portable
    - Edge inference when traveling

MacBook Pro 16" M4 Pro:
  cpu: 14-core (10P + 4E)
  gpu: 20-core
  neural_engine: 16-core
  unified_memory: 48GB
  storage: 1TB SSD
  display: 16" Liquid Retina XDR
  power: 140W USB-C adapter
  qty: 1
  role: "Executive mobile workstation, presentations, client meetings"
  
  capabilities:
    - Ollama (70B models possible, 22B recommended for battery)
    - ComfyUI portable (SDXL generation)
    - LMStudio for quick model testing
    - Full orchestration stack (Docker Desktop)
    - Tailscale for remote PAIS access

Dell Pro Max 16 Plus:
  cpu: Intel Core Ultra 9 285HX (24-core, 2.8 GHz)
  gpu: NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU
  memory: 32GB DDR5
  storage: 2TB NVMe
  display: 16" (specs vary by config)
  qty: 1
  role: "GPU-accelerated mobile workstation, training on-the-go"
  
  capabilities:
    - Ollama with GPU acceleration
    - Fine-tuning small models (7B-13B LoRA)
    - ComfyUI with full SDXL performance
    - vLLM server for local demos
    - Windows-specific AI tools
```

#### Edge Layer

```yaml
Jetson Orin Nano Super:
  cuda_cores: 1024
  memory: 8GB LPDDR5
  power: 15W
  qty: 3
  role: "Edge inference, STT/TTS, IoT integration"
  
  inference_engine: Ollama
  models:
    - Qwen 2.5 7B (Q4_K_M)
    - Whisper Tiny/Base
    - Piper TTS

Raspberry Pi 5:
  cpu: "Broadcom BCM2712 (quad-core Arm Cortex-A76, 2.4 GHz)"
  memory: 16GB LPDDR4X
  power: 10W
  qty: 3
  role: "Home automation, monitoring, lightweight tasks"
```

#### Storage Layer

```yaml
Synology DS423 NAS:
  disks: "4× 6TB IronWolf Pro"
  raid: "SHR (Synology Hybrid RAID) + Btrfs"
  total_capacity: 24TB usable
  power: ~45W
  qty: 1
  
  volumes:
    /volume1/models: "Ollama models, embedding models (~3TB)"
    /volume1/qdrant: "Vector database snapshots"
    /volume1/docker: "Docker registry, container images"
    /volume1/git: "Source code mirrors (Gitea backup)"
    /volume1/backups: "System backups and snapshots"
    /volume1/family: "Family documents + encrypted cloud backup"
    /volume1/synthetic: "AI-generated outputs"
    /volume1/openclaw: "OpenClaw persistent memory and logs"
```

### 3.2 Architecture Overview

The PAIS architecture consists of six logical layers connected via Tailscale mesh networking:

**Layer 1: Orchestration** (Mac mini M4 Pro)
- LangFlow, LMStudio, AnythingLLM, n8n
- API Gateway (Traefik), Model Router, Governance Engine
- Prometheus & Grafana observability

**Layer 2: Local Inference** (Ollama Primary)
- Mac Studio M3 Ultra: Interactive 70B-236B inference, ComfyUI
- DGX Spark #1: Batch inference, high-concurrency
- DGX Spark #2: Fine-tuning, LoRA training
- Jetson Orin (×3): Edge 7B inference, STT/TTS

**Layer 3: OpenClaw Personal Assistant** (GMKtec K6)
- OpenClaw 24/7 proactive AI assistant
- Ollama 22B model (Mistral Small or Qwen 2.5 14B)
- Email, calendar, file management, system monitoring
- Secondary orchestration and development

**Layer 4: Cloud Overflow** (When Local Insufficient)
- NVIDIA Brev: 405B+ models, GPU-intensive workloads
- Google Vertex AI / AI Studio: Enterprise inference
- Gemini API: Fast, cost-effective inference
- ElevenLabs: Professional voice generation
- *Automatic PII redaction before cloud requests*

**Layer 5: Knowledge** (AI Max+ 395)
- Qdrant vector database (64GB allocation)
- Redis semantic cache (32GB allocation)
- Embedding generation, document ingestion

**Layer 6: Storage** (Synology DS423, 24TB)
- Docker registry, Ollama models, backups
- Gitea mirror, family documents, OpenClaw data

**Mobile Tier: Workstations** (ThinkPad, MacBook Pro, Dell Pro Max)
- On-the-go AI inference and development
- Remote access to PAIS via Tailscale
- Portable orchestration and model testing

*See accompanying draw.io diagram for visual architecture overview.*

---

## 4. Orchestration Layer

The orchestration layer provides the user interface, workflow automation, and service coordination for the entire PAIS system. Version 1.3.2 embraces flexibility, supporting multiple orchestration tools that can work independently or in concert.

### 4.1 Supported Orchestration Tools

#### LangFlow: Visual Workflow Builder

```yaml
langflow:
  image: langflow/langflow:latest
  container_name: langflow
  ports:
    - "7860:7860"
  volumes:
    - langflow-data:/root/.langflow
  environment:
    - DATABASE_URL=postgresql://langflow:password@postgres:5432/langflow
    - LANGFLOW_LOG_LEVEL=INFO
    - REDIS_URL=redis://ai-max-395.tailnet.ts.net:6379
  restart: always
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:7860/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

**Best for**:
- Non-technical family members creating AI workflows
- Visual RAG pipeline construction
- Testing prompt chains before production
- Combining LLMs with external APIs (weather, news, etc.)

#### LMStudio: Local Model Testing & Server

```yaml
lmstudio:
  image: custom-lmstudio-server:latest
  container_name: lmstudio
  ports:
    - "8080:8080"   # REST API
    - "1234:1234"   # LMStudio server
  volumes:
    - /Volumes/models:/models:ro
  environment:
    - MODELS_PATH=/models
    - SERVER_PORT=8080
  restart: always
```

**Best for**:
- Testing new models before production deployment
- A/B testing different quantizations
- Local model serving with OpenAI-compatible API

#### AnythingLLM: RAG-Focused Chat Interface

```yaml
anythingllm:
  image: mintplexlabs/anythingllm:latest
  container_name: anythingllm
  ports:
    - "3001:3001"
  volumes:
    - anythingllm-storage:/app/server/storage
    - /Volumes/models:/models:ro
  environment:
    - LLM_PROVIDER=ollama
    - OLLAMA_BASE_PATH=http://mac-studio.tailnet.ts.net:11434
    - EMBEDDING_ENGINE=ollama
    - EMBEDDING_MODEL_PREF=nomic-embed-text
    - VECTORDB=qdrant
    - QDRANT_ENDPOINT=http://ai-max-395.tailnet.ts.net:6333
  restart: always
```

**Best for**:
- Family chat interface with RAG
- Multi-workspace management
- Document upload and automatic indexing

#### n8n: Automation and Integration Workflows

```yaml
n8n:
  image: n8nio/n8n:latest
  container_name: n8n
  ports:
    - "5678:5678"
  volumes:
    - n8n-data:/home/node/.n8n
  environment:
    - N8N_HOST=mac-mini.tailnet.ts.net
    - DB_TYPE=postgres
  restart: always
```

**Best for**:
- Scheduling recurring AI tasks
- Connecting AI with external services
- Multi-step workflows with conditional logic

### 4.2 Service URLs

| Tool | URL | Purpose |
|------|-----|---------|
| **LangFlow** | `http://mac-mini.tailnet.ts.net:7860` | Visual workflow builder |
| **LMStudio** | `http://mac-mini.tailnet.ts.net:8080` | Model testing API |
| **AnythingLLM** | `http://mac-mini.tailnet.ts.net:3001` | Chat interface |
| **n8n** | `http://mac-mini.tailnet.ts.net:5678` | Automation workflows |
| **OpenClaw** | `http://k6.tailnet.ts.net:3000` | Personal AI assistant |

---

## 5. Inference Architecture

### 5.1 Ollama as Primary Inference Engine

PAIS uses **Ollama** as the default inference engine across all local hardware. Ollama provides a simple, unified API for running open-source LLMs.

#### Why Ollama?

- **Simple API**: OpenAI-compatible endpoints
- **Easy model management**: `ollama pull`, `ollama run`, `ollama list`
- **Native acceleration**: Metal (Mac), CUDA (NVIDIA), ROCm (AMD)
- **Model library**: Access to Llama, Qwen, Mistral, DeepSeek, and more
- **Streaming support**: Real-time token generation
- **Multi-modal**: Vision models (LLaVA), code models (CodeLlama)

#### Installing Ollama

**Mac Studio / Mac mini / MacBook Pro:**
```bash
# Install via Homebrew
brew install ollama

# Or download from ollama.com
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve
```

**GMKtec K6 / ThinkPad / Linux:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Run as Docker container
docker run -d \
  --gpus all \
  -v /volume1/models:/root/.ollama \
  -p 11434:11434 \
  --name ollama \
  ollama/ollama
```

**Jetson Orin (ARM64):**
```bash
# ARM64 installation
curl -fsSL https://ollama.com/install.sh | sh

# Or Docker
docker run -d \
  --runtime nvidia \
  -v /opt/ollama:/root/.ollama \
  -p 11434:11434 \
  --name ollama \
  ollama/ollama
```

### 5.2 Model Management with Ollama

#### Pulling Models

```bash
# Pull a model from Ollama library
ollama pull llama3.2:70b

# Pull quantized versions
ollama pull llama3.2:70b-q4_K_M
ollama pull qwen2.5:72b-q4_K_M
ollama pull deepseek-v3:236b-q4_K_M  # New with Mac Studio 256GB

# List installed models
ollama list

# Remove models
ollama rm llama3.2:70b
```

#### Model Storage

Ollama stores models in:
- **Mac**: `~/.ollama/models`
- **Linux**: `/usr/share/ollama/.ollama/models`
- **Docker**: Volume mount (e.g., `/volume1/models`)

**Shared model storage via NAS:**

```bash
# Symlink Ollama models to NAS
ln -s /Volumes/models/ollama ~/.ollama/models

# Or mount NAS directly
export OLLAMA_MODELS=/Volumes/models/ollama
ollama serve
```

### 5.3 Running Ollama on Different Hardware

#### Mac Studio M3 Ultra (Primary Interactive Inference)

```bash
# Start Ollama server
ollama serve

# Run 70B model interactively
ollama run llama3.2:70b

# NEW: Run 236B model with 256GB RAM
ollama run deepseek-v3:236b-q4_K_M
```

**Performance expectations:**

| Model | Quantization | Tokens/sec | TTFT | Memory |
|-------|--------------|------------|------|--------|
| Llama 3.2 70B | Q4_K_M | 6-7 | 8s | ~45GB |
| Qwen 2.5 72B | Q4_K_M | 5-6 | 10s | ~48GB |
| DeepSeek v3 236B | Q4_K_M | 2-3 | 25s | ~155GB |
| Mistral Small 22B | Q4_K_M | 15-20 | 3s | ~15GB |
| DeepSeek Coder 33B | Q4_K_M | 10-12 | 5s | ~22GB |

#### GMKtec K6 (OpenClaw + Dev)

```bash
# Run 22B model for OpenClaw
ollama run mistral-small:latest

# Or smaller for faster response
ollama run qwen2.5:14b
```

**Performance**: 10-15 tok/s for 22B models on CPU

#### Mobile Workstations

**ThinkPad P14s (96GB RAM):**
```bash
# Can run up to 70B models on CPU
ollama run llama3.2:70b  # ~3-5 tok/s on CPU

# Recommended: 22B for battery efficiency
ollama run mistral-small:latest  # ~8-12 tok/s
```

**MacBook Pro M4 Pro (48GB RAM):**
```bash
# Optimal: 22B models with Metal acceleration
ollama run mistral-small:latest  # ~15-20 tok/s with Metal

# Possible: 70B models (slow, battery drain)
ollama run llama3.2:70b  # ~3-4 tok/s
```

**Dell Pro Max 16 Plus (RTX 5000 Blackwell):**
```bash
# GPU-accelerated inference
ollama run llama3.2:70b  # ~20-30 tok/s with GPU

# Recommended for presentations
ollama run qwen2.5:32b  # ~50-70 tok/s
```

#### DGX Spark (Batch Inference)

```yaml
# docker-compose.ollama-dgx.yml
services:
  ollama:
    image: ollama/ollama
    container_name: ollama-dgx
    ports:
      - "11434:11434"
    volumes:
      - /volume1/models/ollama:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - OLLAMA_NUM_PARALLEL=4
      - OLLAMA_MAX_LOADED_MODELS=2
    restart: always
```

#### Jetson Orin (Edge Inference)

```bash
# Run 7B model on Jetson
ollama run qwen2.5:7b

# Or smaller models
ollama run phi3:mini      # 3.8B parameters
ollama run llama3.2:3b
```

**Performance**: 8-15 tok/s for 7B models

### 5.4 Ollama API Usage

#### OpenAI-Compatible API (Pseudocode)

```
# Pseudocode: OpenAI-compatible API client

function ollama_chat_completion(model, messages):
  endpoint = "http://mac-studio.tailnet.ts.net:11434/v1/chat/completions"
  
  payload = {
    "model": model,
    "messages": messages,
    "stream": false
  }
  
  response = http_post(endpoint, payload)
  
  return response.choices[0].message.content


# Example usage
messages = [
  {"role": "user", "content": "Explain quantum computing"}
]

result = ollama_chat_completion("llama3.2:70b", messages)
print(result)
```

#### Native Ollama API (Pseudocode)

```
# Pseudocode: Native Ollama API

function ollama_generate(model, prompt):
  endpoint = "http://mac-studio.tailnet.ts.net:11434/api/generate"
  
  payload = {
    "model": model,
    "prompt": prompt,
    "stream": false
  }
  
  response = http_post(endpoint, payload)
  
  return response.response


# Example usage
result = ollama_generate("llama3.2:70b", "Explain quantum computing")
print(result)
```

#### Streaming Responses (Pseudocode)

```
# Pseudocode: Streaming responses

function ollama_generate_stream(model, prompt):
  endpoint = "http://mac-studio.tailnet.ts.net:11434/api/generate"
  
  payload = {
    "model": model,
    "prompt": prompt,
    "stream": true
  }
  
  # Open streaming connection
  stream = http_post_stream(endpoint, payload)
  
  for chunk in stream:
    if chunk:
      data = json_parse(chunk)
      print(data.response, end="", flush=true)


# Example usage
ollama_generate_stream("llama3.2:70b", "Write a story about AI")
```

### 5.5 ComfyUI on Mac Studio

ComfyUI runs natively on macOS with MPS (Metal Performance Shaders):

```yaml
# Models stored on NAS, mounted via NFS
# ~/ComfyUI/extra_model_paths.yaml
pais_nas:
  base_path: /Volumes/models
  checkpoints: stable-diffusion/
  loras: loras/
  vae: vae/
  controlnet: controlnet/
```

**Mount NAS on Mac:**

```bash
# Mount NAS via NFS with macOS compatibility
mkdir -p ~/NAS/models
mount_nfs -o resvport,nfc ds423.tailnet.ts.net:/volume1/models ~/NAS/models
```

**Performance**: SDXL generation ~25-35s per image (1024×1024, 30 steps)

### 5.6 Model Quantization

```bash
# Ollama automatically downloads quantized models
ollama pull llama3.2:70b        # Default (Q4_K_M)
ollama pull llama3.2:70b-q8_0   # Higher quality (INT8)
ollama pull llama3.2:70b-q4_0   # More compression (INT4)
```

**Quantization trade-offs:**

| Quantization | VRAM Reduction | Quality Impact | Best For |
|--------------|----------------|----------------|----------|
| **Q8_0** | ~50% | 1-2% accuracy loss | High-quality inference |
| **Q4_K_M** | ~75% | 2-4% accuracy loss | General use (default) |
| **Q4_0** | ~75% | 3-5% accuracy loss | Maximum compression |
| **Q2_K** | ~85% | 5-10% accuracy loss | Experimental |

---

## 6. OpenClaw Personal AI Assistant

OpenClaw is an open-source personal AI assistant that provides 24/7 proactive monitoring and task automation. Unlike passive chatbots, OpenClaw actively monitors your systems, sends heartbeat updates, and performs tasks autonomously.

### 6.1 Why OpenClaw?

**Capabilities:**
- **Email management**: Monitor inbox, send automated responses, schedule follow-ups
- **Calendar integration**: Schedule meetings, send reminders, manage conflicts
- **File organization**: Automatically organize downloads, documents, photos
- **System monitoring**: Monitor server health, disk space, backups
- **Browser automation**: Fill forms, scrape data, automate workflows
- **Proactive assistance**: "It's 9 AM, here's your schedule and unread emails"
- **Messaging integration**: Telegram, Slack, WhatsApp notifications

**Local-first design:**
- Runs on Ollama (no mandatory cloud APIs)
- Optional cloud fallback for complex reasoning
- Persistent memory stored locally
- Docker-based for easy deployment

### 6.2 Deployment on GMKtec K6

#### Why K6 is Perfect for OpenClaw

✅ **Always-on capability**: Low power (~45W), dedicated machine  
✅ **64GB RAM**: Sufficient for OpenClaw + local 22B model  
✅ **Dual NIC**: Network redundancy for 24/7 operation  
✅ **Separate from critical services**: Won't interfere with orchestration/inference layers  
✅ **Local execution**: Keeps OpenClaw's system access isolated  
✅ **USB4**: Fast NAS access for file operations

#### Docker Compose Configuration

```yaml
# docker-compose.openclaw.yml on K6
version: '3.8'

services:
  openclaw:
    image: openclaw/openclaw:latest
    container_name: openclaw
    ports:
      - "3000:3000"   # Web UI
      - "8765:8765"   # WebSocket
    volumes:
      - openclaw-data:/root/.openclaw
      - /home/user:/host_home:rw  # For file operations
      - /volume1/openclaw:/persistent:rw  # NAS storage for long-term memory
    environment:
      # AI Model Configuration
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}  # Optional cloud fallback
      - OPENAI_API_KEY=${OPENAI_API_KEY}  # Optional cloud fallback
      - OLLAMA_BASE_URL=http://localhost:11434
      - DEFAULT_MODEL=mistral-small:latest
      
      # Messaging Integrations
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - SLACK_TOKEN=${SLACK_TOKEN}
      - WHATSAPP_ENABLED=true
      
      # Permissions (start restrictive, expand as needed)
      - PERMISSIONS_SHELL=true
      - PERMISSIONS_BROWSER=true
      - PERMISSIONS_FILESYSTEM=true
      - PERMISSIONS_NETWORK=true
      
      # Heartbeat (proactive monitoring)
      - HEARTBEAT_ENABLED=true
      - HEARTBEAT_INTERVAL=5m
      - HEARTBEAT_RECIPIENTS=telegram,slack
      
      # Memory
      - MEMORY_ENABLED=true
      - MEMORY_MAX_CONTEXT=50000
      - MEMORY_STORAGE=/persistent/memory
      
      # Security
      - ADMIN_PASSWORD=${OPENCLAW_ADMIN_PASSWORD}
      - JWT_SECRET=${OPENCLAW_JWT_SECRET}
    restart: always
    depends_on:
      - ollama
      - postgres

  ollama:
    image: ollama/ollama
    container_name: ollama-openclaw
    ports:
      - "11434:11434"
    volumes:
      - /volume1/models/ollama:/root/.ollama
    environment:
      - OLLAMA_NUM_PARALLEL=2
      - OLLAMA_MAX_LOADED_MODELS=1
    restart: always

  postgres:
    image: postgres:15-alpine
    container_name: openclaw-db
    environment:
      - POSTGRES_DB=openclaw
      - POSTGRES_USER=openclaw
      - POSTGRES_PASSWORD=${OPENCLAW_DB_PASSWORD}
    volumes:
      - openclaw-db:/var/lib/postgresql/data
    restart: always

volumes:
  openclaw-data:
  openclaw-db:
```

#### Initial Setup

```bash
# On GMKtec K6

# 1. Pull Ollama model for OpenClaw
ollama pull mistral-small:latest  # 22B, excellent for assistant tasks
# or
ollama pull qwen2.5:14b  # 14B, faster, still capable

# 2. Create environment file
cat > .env << EOF
OPENCLAW_ADMIN_PASSWORD=your_secure_password
OPENCLAW_JWT_SECRET=$(openssl rand -hex 32)
OPENCLAW_DB_PASSWORD=$(openssl rand -hex 16)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
SLACK_TOKEN=your_slack_token
ANTHROPIC_API_KEY=optional_claude_api_key
OPENAI_API_KEY=optional_openai_api_key
EOF

# 3. Start OpenClaw
docker-compose -f docker-compose.openclaw.yml up -d

# 4. Access web UI
# http://k6.tailnet.ts.net:3000
```

### 6.3 OpenClaw Use Cases

#### Use Case 1: Email Monitoring and Response

```yaml
# OpenClaw task configuration
task:
  name: "Email Triage"
  schedule: "*/15 * * * *"  # Every 15 minutes
  actions:
    - monitor_inbox:
        filters:
          - unread: true
          - priority: high
    - categorize:
        categories: [urgent, work, personal, spam]
    - auto_respond:
        condition: "category == 'urgent' and from_domain == 'work.com'"
        template: "I've received your email and will respond within 2 hours."
    - notify:
        channel: telegram
        message: "{{ unread_count }} unread emails, {{ urgent_count }} urgent"
```

#### Use Case 2: System Health Monitoring

```yaml
task:
  name: "PAIS Health Check"
  schedule: "0 * * * *"  # Hourly
  actions:
    - check_services:
        endpoints:
          - http://mac-studio.tailnet.ts.net:11434/api/tags  # Ollama
          - http://ai-max-395.tailnet.ts.net:6333/health  # Qdrant
          - http://mac-mini.tailnet.ts.net:3001/health  # AnythingLLM
    - check_disk_space:
        warning_threshold: 80%
        critical_threshold: 90%
    - notify_on_failure:
        channel: telegram
        message: "⚠️ {{ service }} is down!"
```

#### Use Case 3: Daily Briefing

```yaml
task:
  name: "Morning Briefing"
  schedule: "0 8 * * *"  # 8 AM daily
  actions:
    - fetch_calendar:
        timeframe: today
    - fetch_emails:
        filters:
          - unread: true
          - date: today
    - fetch_weather:
        location: Atlanta, GA
    - generate_summary:
        prompt: |
          Generate a concise morning briefing:
          - Today's schedule: {{ calendar_events }}
          - Unread emails: {{ email_count }}
          - Weather: {{ weather_summary }}
    - send_message:
        channel: telegram
        message: "{{ summary }}"
```

#### Use Case 4: File Organization

```yaml
task:
  name: "Organize Downloads"
  schedule: "0 0 * * *"  # Daily at midnight
  actions:
    - scan_directory:
        path: /home/user/Downloads
    - categorize_files:
        rules:
          - type: pdf
            destination: /home/user/Documents/PDFs
          - type: image
            destination: /home/user/Pictures
          - type: video
            destination: /home/user/Videos
          - type: archive
            destination: /home/user/Archives
    - move_files:
        confirm: false
    - notify:
        channel: slack
        message: "Organized {{ file_count }} files from Downloads"
```

### 6.4 Integration with PAIS Orchestration

#### Routing Tasks Between OpenClaw and n8n (Pseudocode)

```
# Pseudocode: TaskRouter

class TaskRouter:
  
  OPENCLAW_TASKS = [
    "email", "calendar", "schedule", "organize files",
    "monitor", "remind", "health check", "browser automation"
  ]
  
  N8N_TASKS = [
    "data pipeline", "ETL", "API integration",
    "scheduled batch", "webhook", "complex workflow"
  ]
  
  
  function route(task_description):
    task_lower = to_lowercase(task_description)
    
    # OpenClaw: system integration, proactive monitoring
    if any_keyword_in(OPENCLAW_TASKS, task_lower):
      return "http://k6.tailnet.ts.net:3000/api/task"
    
    # n8n: data workflows, API integrations
    else if any_keyword_in(N8N_TASKS, task_lower):
      return "http://mac-mini.tailnet.ts.net:5678/webhook/task"
    
    # Default: OpenClaw for general assistance
    else:
      return "http://k6.tailnet.ts.net:3000/api/task"
```

### 6.5 Security Considerations

#### Permissions Management

OpenClaw requires system-level permissions for full functionality. Use least-privilege approach:

```yaml
# Start with minimal permissions
environment:
  - PERMISSIONS_SHELL=false
  - PERMISSIONS_BROWSER=false
  - PERMISSIONS_FILESYSTEM=read_only
  - PERMISSIONS_NETWORK=true

# Gradually expand based on tasks
# Example: Enable shell for system monitoring
environment:
  - PERMISSIONS_SHELL=true
  - SHELL_ALLOWED_COMMANDS=["df", "free", "systemctl status"]
```

#### Access Control via Tailscale

```json
{
  "acls": [
    {
      "action": "accept",
      "src": ["group:admins"],
      "dst": ["tag:openclaw:3000", "tag:openclaw:8765"]
    },
    {
      "action": "deny",
      "src": ["group:family"],
      "dst": ["tag:openclaw:*"]
    }
  ]
}
```

**Rationale**: OpenClaw has system access; restrict to admins only

---

## 7. Cloud Overflow Strategy

PAIS is **local-first** but pragmatic. When local resources are insufficient—model too large, GPU memory exhausted, high concurrency—intelligently route to cloud services.

### 7.1 Cloud Service Providers

#### NVIDIA Brev (Pseudocode)

**Use case**: Large model inference (405B+), GPU-intensive workloads

```
# Pseudocode: NVIDIA Brev client

function brev_inference(prompt, model = "llama-3.3-405b-instruct"):
  api_key = get_env_var("BREV_API_KEY")
  endpoint = "https://api.brev.dev/v1/inference"
  
  headers = {
    "Authorization": "Bearer " + api_key,
    "Content-Type": "application/json"
  }
  
  payload = {
    "model": model,
    "messages": [
      {"role": "user", "content": prompt}
    ],
    "max_tokens": 1024
  }
  
  response = http_post(endpoint, payload, headers)
  
  return response.choices[0].message.content


# Example usage
result = brev_inference("Explain quantum entanglement", "llama-3.3-405b-instruct")
print(result)
```

**When to use**: 180B+ models not practical locally, need cutting-edge performance

#### Google Vertex AI (Pseudocode)

**Use case**: Enterprise-grade inference, fine-tuning, multi-modal

```
# Pseudocode: Google Vertex AI client

function vertex_inference(prompt, model = "text-bison@002"):
  project_id = "your-project-id"
  location = "us-central1"
  
  # Initialize Vertex AI client
  vertex_client = initialize_vertex_ai(project_id, location)
  
  # Get pre-trained model
  text_model = vertex_client.get_text_generation_model(model)
  
  # Generate prediction
  response = text_model.predict(
    prompt = prompt,
    max_output_tokens = 1024,
    temperature = 0.7
  )
  
  return response.text


# Example usage
result = vertex_inference("Summarize quantum computing applications")
print(result)
```

#### Gemini API (Pseudocode)

**Use case**: Fast, cost-effective inference for general tasks

```
# Pseudocode: Gemini API client

function gemini_inference(prompt, model = "gemini-1.5-flash"):
  api_key = get_env_var("GEMINI_API_KEY")
  
  # Configure Gemini client
  gemini_client = initialize_gemini(api_key)
  
  # Get generative model
  generative_model = gemini_client.get_model(model)
  
  # Generate content
  response = generative_model.generate_content(prompt)
  
  return response.text


# Example usage
result = gemini_inference("What is artificial general intelligence?")
print(result)
```

#### ElevenLabs Audio (Pseudocode)

**Use case**: Professional voice generation, voice cloning, TTS

```
# Pseudocode: ElevenLabs TTS client

function elevenlabs_tts(text, voice = "Rachel"):
  api_key = get_env_var("ELEVENLABS_API_KEY")
  
  # Initialize ElevenLabs client
  elevenlabs_client = initialize_elevenlabs(api_key)
  
  # Generate audio
  audio_bytes = elevenlabs_client.generate(
    text = text,
    voice = voice,
    model = "eleven_multilingual_v2"
  )
  
  return audio_bytes


# Example usage
audio = elevenlabs_tts("Hello, welcome to PAIS", "Rachel")
save_audio_file(audio, "welcome.mp3")
```

### 7.2 Intelligent Routing Logic (Pseudocode)

```
# Pseudocode: CloudRouter

enum InferenceTarget:
  LOCAL_OLLAMA
  CLOUD_BREV
  CLOUD_VERTEX
  CLOUD_GEMINI


class CloudRouter:
  
  function __init__():
    this.pii_detector = PIIDetector()
  
  
  function route(prompt, model, max_tokens = 512):
    # Check if model is too large for local
    if this.is_large_model(model):
      return InferenceTarget.CLOUD_BREV
    
    # Check local GPU availability
    if not this.has_local_capacity():
      return InferenceTarget.CLOUD_GEMINI
    
    # Default: use local Ollama
    return InferenceTarget.LOCAL_OLLAMA
  
  
  function is_large_model(model):
    # Mac Studio can handle up to 236B with 256GB RAM
    large_models = ["405b", "claude", "gpt-4"]
    
    for large_model in large_models:
      if large_model in to_lowercase(model):
        return true
    
    return false
  
  
  function execute(target, prompt, model, redact_pii = true):
    # Always redact PII before cloud
    if target != InferenceTarget.LOCAL_OLLAMA and redact_pii:
      if not this.pii_detector.is_safe(prompt):
        prompt = this.pii_detector.redact(prompt)
    
    if target == InferenceTarget.LOCAL_OLLAMA:
      return this.ollama_inference(prompt, model)
    else if target == InferenceTarget.CLOUD_BREV:
      return brev_inference(prompt, model)
    else if target == InferenceTarget.CLOUD_GEMINI:
      return gemini_inference(prompt)
  
  
  function ollama_inference(prompt, model):
    endpoint = "http://mac-studio.tailnet.ts.net:11434/api/generate"
    
    payload = {
      "model": model,
      "prompt": prompt,
      "stream": false
    }
    
    response = http_post(endpoint, payload)
    
    return response.response
```

---

## 7.3 PII Detection (Nemotron Content Safety)

### 7.3.1 Overview

Section 7.3 uses **NVIDIA Nemotron-4 Content Safety Reasoning** as the core mechanism for detecting and handling PII before any request is sent to cloud providers. All examples are **pseudocode** to remain language-agnostic; infrastructure snippets remain **YAML**.

Nemotron replaces previous regex-based detection and provides:

- Context-aware PII detection ("my SSN is ..." vs random digits)
- Natural-language reasoning about why content is flagged
- Redacted text suitable for safe cloud use
- Consistent API for use across CloudRouter, OpenClaw, n8n, and RAG

---

### 7.3.2 Deployment Architecture

Nemotron PII runs on the **AI Max+ 395** with Redis caching. A Jetson Orin node can serve as fallback.

```yaml
# docker-compose.nemotron-pii.yml on AI Max+ 395
version: '3.8'

services:
  nemotron-pii:
    image: nvcr.io/nvidia/nemotron-4-340b-content-safety-reasoning:latest
    container_name: nemotron-pii
    ports:
      - "8888:8000"  # vLLM API endpoint
    volumes:
      - /volume1/models/nemotron:/model_cache
      - /tmp/nemotron_logs:/logs
    environment:
      # Model configuration
      - MODEL_NAME=nemotron-4-340b-content-safety-reasoning
      - QUANTIZATION=awq  # 4-bit quantization
      - GPU_MEMORY_UTILIZATION=0.7
      
      # API configuration
      - VLLM_PORT=8000
      - MAX_CONCURRENT_REQUESTS=128
      - MAX_MODEL_LEN=2048
      
      # Logging
      - LOG_LEVEL=INFO
      - LOG_PATH=/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis cache for repeated requests
  pii-cache:
    image: redis:7-alpine
    container_name: pii-cache
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 4gb --maxmemory-policy allkeys-lru
    volumes:
      - pii-cache-data:/data
    restart: always

volumes:
  pii-cache-data:
```

High-level placement:

- **Primary**: AI Max+ 395 (Nemotron PII)
- **Fallback**: Jetson Orin Nano (quantized / distilled PII model)
- **Consumers**: CloudRouter, OpenClaw, AnythingLLM, n8n, RAG pipeline

---

### 7.3.3 Nemotron PII Detection API (Conceptual)

Nemotron exposes a JSON HTTP endpoint.

```yaml
# POST /v1/content-safety/detect

request:
  text: "User input text to analyze"
  detect_pii: true
  detect_hate_speech: false
  detect_violence: false
  detect_sexual_content: false
  languages: ["en"]
  model: "nemotron-4-content-safety-reasoning"
  
response:
  violations:
    - type: "pii.ssn"
      severity: "high"
      confidence: 0.98
      detected_value: "[REDACTED]"
      reasoning: "Pattern matches US Social Security Number format with contextual cues"
      position: {start: 11, end: 23}
  
  safe_for_cloud: false
  redacted_text: "My SSN is [REDACTED]"
  processing_time_ms: 187
```

Representative PII types:

- `pii.ssn`, `pii.credit_card`, `pii.email`, `pii.phone`, `pii.ip_address`
- `pii.passport`, `pii.driver_license`, `pii.bank_account`
- `pii.api_key`, `pii.access_token`
- `pii.dob`, `pii.address`, `pii.medical`

---

### 7.3.4 NemotronPIIDetector (Pseudocode)

```
# Pseudocode: Nemotron PII Detector

class NemotronPIIDetector:
  
  function __init__(nemotron_base_url, cache_client):
    this.base_url = nemotron_base_url   # "http://ai-max-395.tailnet.ts.net:8888"
    this.cache = cache_client           # Redis or in-memory cache
  
  
  function detect(text):
    # 1. Cache lookup
    cache_key = "pii:" + hash(text)
    cached = this.cache.get(cache_key)
    
    if cached is not null:
      return deserialize(cached)
    
    # 2. Call Nemotron API
    payload = {
      "text": text,
      "detect_pii": true,
      "languages": ["en"]
    }
    
    response = http_post(this.base_url + "/v1/content-safety/detect", payload)
    
    # 3. Cache result with TTL
    this.cache.set(cache_key, serialize(response), ttl_seconds = 86400)
    
    return response
  
  
  function is_safe_for_cloud(text):
    result = this.detect(text)
    return result.safe_for_cloud == true
  
  
  function redact(text):
    result = this.detect(text)
    
    if result.redacted_text is not null:
      return result.redacted_text
    else:
      return text
```

---

### 7.3.5 CloudRouter with Nemotron (Pseudocode)

```
# Pseudocode: CloudRouter with Nemotron PII filtering

class CloudRouterWithNemotron:
  
  function __init__(pii_detector):
    this.pii_detector = pii_detector
  
  
  function route(prompt, model, max_tokens):
    # Determine inference target (local vs cloud)
    if this.is_large_model(model):
      return CLOUD_BREV
    
    if not this.has_local_capacity():
      return CLOUD_GEMINI
    
    return LOCAL_OLLAMA
  
  
  function execute(target, prompt, model, redact_pii = true):
    # Only PII-check non-local targets
    if target != LOCAL_OLLAMA and redact_pii == true:
      result = this.pii_detector.detect(prompt)
      
      if result.safe_for_cloud == false:
        # Apply policy (strict/moderate/permissive)
        policy = get_pii_policy_for_target(target)
        
        if policy == "strict":
          # Block and surface to user / logs
          log_pii_violation(target, model, result)
          raise Error("Cloud call blocked due to PII")
        
        if policy == "moderate":
          # Auto-redact then continue
          prompt = result.redacted_text
          log_pii_violation(target, model, result)
        
        if policy == "permissive":
          # Log but still allow
          log_pii_violation(target, model, result)
    
    return this.execute_on_target(target, prompt, model)
  
  
  function execute_on_target(target, prompt, model):
    if target == LOCAL_OLLAMA:
      return ollama_inference(prompt, model)
    else if target == CLOUD_BREV:
      return brev_inference(prompt, model)
    else if target == CLOUD_GEMINI:
      return gemini_inference(prompt)
    else if target == CLOUD_VERTEX:
      return vertex_inference(prompt, model)
```

---

### 7.3.6 OpenClawPIIFilter (Pseudocode)

```
# Pseudocode: OpenClaw PII Filter

class OpenClawPIIFilter:
  
  function __init__(pii_detector, alerting_system):
    this.pii_detector = pii_detector
    this.alerts = alerting_system
  
  
  function process_task(task):
    cloud_like = ["email_sync", "calendar_sync", "send_message", "webhook"]
    
    if task.type not in cloud_like:
      return task
    
    for (name, value) in task.parameters:
      if type(value) == "string" and length(value) > 0:
        result = this.pii_detector.detect(value)
        
        if length(result.violations) > 0:
          policy = get_pii_policy_for("openclaw")
          
          if policy == "strict":
            task.blocked = true
            this.alerts.send(
              "OpenClaw task blocked due to PII",
              {"task_type": task.type, "violations": result.violations}
            )
            return task
          
          if policy == "moderate":
            task.parameters[name] = result.redacted_text
            this.alerts.send(
              "OpenClaw task auto-redacted PII",
              {"task_type": task.type, "violations": result.violations}
            )
    
    return task
```

---

### 7.3.7 n8n Workflow Integration (YAML)

```yaml
pii_check_node:
  displayName: "Check for PII (Nemotron)"
  type: "http"
  typeVersion: 4.1
  position: [400, 300]
  
  parameters:
    url: "http://ai-max-395.tailnet.ts.net:8888/v1/content-safety/detect"
    method: "POST"
    authentication: "none"
    sendHeaders: true
    headerParameters:
      parameters:
        - name: "Content-Type"
          value: "application/json"
    sendBody: true
    bodyParameters:
      raw: |
        {
          "text": "{{ $node[\"Previous Node\"].json.user_input }}",
          "detect_pii": true,
          "languages": ["en"]
        }
    responseFormat: "json"
  
  outputs:
    mainOutputs:
      - outputName: "success"
        outputs:
          - nodeOutputType: "json"
            outputProperties:
              - name: "violations"
              - name: "safe_for_cloud"
              - name: "redacted_text"
              - name: "processing_time_ms"

conditional_routing:
  - condition: "$node['pii_check_node'].json.safe_for_cloud == true"
    route_to: "cloud_inference_node"
  
  - condition: "$node['pii_check_node'].json.safe_for_cloud == false"
    route_to: "redact_and_alert_node"
```

---

### 7.3.8 Governance & Policy (YAML)

```yaml
pii_detection_policy:
  enabled: true
  
  detection_mode:
    # "strict": Block unsafe PII, require manual review
    # "moderate": Auto-redact, then allow cloud
    # "permissive": Warn only
    global: "moderate"
    
    per_service:
      gemini: "moderate"
      brev: "strict"
      vertex_ai: "moderate"
      anythingllm: "strict"
      openclaw: "strict"
  
  auto_redaction:
    enabled: true
    pii_types_to_redact:
      - "ssn"
      - "credit_card"
      - "passport"
      - "bank_account"
      - "api_key"
    pii_types_to_warn:
      - "email"
      - "phone"
      - "address"
  
  alerting:
    enabled: true
    channels:
      - type: "telegram"
        user_id: "admin_telegram_id"
      - type: "slack"
        webhook_url: "$SLACK_WEBHOOK_URL"
    
    alert_on_severity: "high"  # Alert on high/critical
  
  logging:
    enabled: true
    retention_days: 90
    locations:
      - "/volume1/logs/pii-violations.jsonl"
      - "prometheus_metrics"

  exceptions:
    allow_pii_bypass: false
    authorized_users: []
```

---

### 7.3.9 Logging & Metrics (Pseudocode / PromQL)

```
# Pseudocode: PII Violation Logger

class PIIViolationLogger:
  
  function log_violation(context, result):
    entry = {
      "timestamp": now(),
      "event_type": "pii.detected",
      "severity": max_severity(result.violations),
      "violation_count": length(result.violations),
      "pii_types": list_types(result.violations),
      "source": context.source,          # "cloud_request", "openclaw", etc.
      "target_service": context.target,  # "gemini", "brev", etc.
      "action_taken": context.action,    # "blocked", "redacted", "allowed"
      "processing_time_ms": result.processing_time_ms
    }
    
    append_jsonl("/volume1/logs/pii-violations.jsonl", entry)
    
    prometheus_counter("pii_violations_total", {
      "severity": entry.severity,
      "pii_type": entry.pii_types[0],
      "source": entry.source
    }).inc()
```

Example Grafana queries:

```promql
# PII violations per hour
rate(pii_violations_total[1h])

# Violations by severity
sum(rate(pii_violations_total[24h])) by (severity)

# Top PII types
topk(5, sum(rate(pii_violations_total[24h])) by (pii_type))

# Blocked vs redacted
sum(rate(pii_violations_total[24h])) by (action)
```

---

### 7.3.10 Fallback & HA (Pseudocode)

```
# Pseudocode: PII Detection Failover

class PIIDetectionFailover:
  
  PRIMARY = "http://ai-max-395.tailnet.ts.net:8888"
  SECONDARY = "http://jetson-orin-1.tailnet.ts.net:8000"
  TIMEOUT_MS = 500
  
  
  function detect(text):
    try:
      return call_with_timeout(
        PRIMARY + "/v1/content-safety/detect",
        text,
        TIMEOUT_MS
      )
    
    catch (TimeoutError, ConnectionError):
      log_warning("Primary PII endpoint unavailable; using secondary")
    
    try:
      return call_with_timeout(
        SECONDARY + "/v1/content-safety/detect",
        text,
        TIMEOUT_MS * 2
      )
    
    catch (Error e):
      log_error("All PII detectors unavailable", e)
      
      # Governance choice: fail-open with logging
      return {
        "safe_for_cloud": true,
        "violations": [],
        "fallback_reason": "pii_detection_unavailable"
      }
```

---

### 7.3.11 Summary of Changes

| Aspect           | Old (Regex-based)            | New (Nemotron-based)                               |
|------------------|------------------------------|----------------------------------------------------|
| Detection method | Regex patterns               | Nemotron-4 Content Safety Reasoning                |
| Accuracy         | ~60–70%                      | Target 95%+                                        |
| Context handling | None                         | Strong (natural-language reasoning)                |
| Examples         | Python                       | Language-agnostic pseudocode + YAML                |
| Deployment       | N/A                          | AI Max+ 395 via vLLM + Redis cache                 |
| Governance       | Ad hoc                       | Central YAML policy + Prometheus/Grafana metrics   |
| Integrations     | Minimal                      | CloudRouter, OpenClaw, AnythingLLM, n8n, RAG       |

---

## 8. Containerization Strategy

Docker containers provide the abstraction layer that makes PAIS truly portable and composable.

### 8.1 Self-Hosted Registry

```yaml
# docker-compose.yml for DS423
services:
  registry:
    image: registry:2
    ports:
      - "5000:5000"
    volumes:
      - /volume1/docker/registry:/var/lib/registry
    restart: always
```

### 8.2 Image Organization

| Registry Path          | Purpose                | Architecture |
|------------------------|------------------------|--------------|
| `pais/langflow`        | Workflow builder       | amd64        |
| `pais/anythingllm`     | RAG chat interface     | amd64        |
| `pais/n8n`             | Automation workflows   | amd64        |
| `pais/openclaw`        | Personal AI assistant  | amd64        |
| `pais/ollama`          | Inference engine       | amd64, arm64 |
| `pais/qdrant`          | Vector database        | amd64        |

### 8.3 Multi-Architecture Builds

```bash
docker buildx create --name pais-builder --use

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ds423.tailnet.ts.net:5000/pais/my-service:v1 \
  --push .
```

---

## 9. Source Code Management

### 9.1 Repository Strategy

| Repository Type | Location              | Purpose            |
|-----------------|----------------------|--------------------|
| Primary         | GitHub (private repos) | Collaboration, CI/CD |
| Mirror          | DS423 NAS (/volume1/git) | Sovereign backup   |
| Working         | Development machine   | Active development |

### 9.2 NAS Git Server

```yaml
services:
  gitea:
    image: gitea/gitea:latest
    ports:
      - "3000:3000"
      - "2222:22"
    volumes:
      - /volume1/git/gitea:/data
    restart: always
```

---

## 10. Observability Stack

### 10.1 Deployment

```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    restart: always

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3002:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    restart: always
```

### 10.2 Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ollama'
    static_configs:
      - targets:
        - 'mac-studio.tailnet.ts.net:11434'
        - 'k6.tailnet.ts.net:11434'

  - job_name: 'openclaw'
    static_configs:
      - targets: ['k6.tailnet.ts.net:3000']

  - job_name: 'qdrant'
    static_configs:
      - targets: ['ai-max-395.tailnet.ts.net:6333']

  - job_name: 'nemotron-pii'
    static_configs:
      - targets: ['ai-max-395.tailnet.ts.net:8888']
```

### 10.3 Key Metrics

```promql
# Ollama tokens per second
rate(ollama_tokens_generated_total[5m])

# OpenClaw tasks per hour
rate(openclaw_tasks_total[1h])

# Cloud API usage
sum(increase(cloud_requests_total[24h])) by (service)

# PII violations
sum(rate(pii_violations_total[24h])) by (severity)

# Nemotron latency (p95)
histogram_quantile(0.95, rate(nemotron_pii_latency_seconds_bucket[5m]))
```

---

## 11. Security and Hardening

### 11.1 Tailscale ACLs

```json
{
  "groups": {
    "group:admins": ["parent@family.tailnet"],
    "group:family": ["parent@family.tailnet", "child1@family.tailnet"]
  },
  "acls": [
    {
      "action": "accept",
      "src": ["group:family"],
      "dst": ["tag:chat:3001"]
    },
    {
      "action": "accept",
      "src": ["group:admins"],
      "dst": ["*:*"]
    }
  ]
}
```

(Additional hardening guidance unchanged from prior version.)

---

## 12. Knowledge Layer

### 12.1 Qdrant on AI Max+ 395

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - /volume1/qdrant:/qdrant/storage
    restart: always

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 32gb
    ports:
      - "6379:6379"
    restart: always
```

---

## 13. Multi-Tenancy and Governance

### 13.1 Per-User Policies

```yaml
users:
  - id: parent@family.tailnet
    role: admin
    allowed_models: ["*"]
    allow_cloud: true
  
  - id: child1@family.tailnet
    role: child
    allowed_models: ["llama3.2:8b"]
    allow_cloud: false
```

---

## 14. Performance Optimization

### 14.1 Tiered Routing (Pseudocode)

```
# Pseudocode: Inference Tier enumeration

enum InferenceTier:
  INTERACTIVE   # Mac Studio (Ollama 70B-236B)
  EDGE          # Jetson (Ollama 7B)
  BATCH         # DGX Spark
  OPENCLAW      # K6 (22B for assistant tasks)
  CLOUD         # NVIDIA Brev, Gemini
```

---

## 15. Backup and Disaster Recovery

### 15.1 3-2-1 Strategy

```yaml
backup_task:
  destination: Backblaze B2
  schedule: Daily 3 AM
  encryption: Client-side AES-256
  included:
    - /volume1/family
    - /volume1/qdrant
    - /volume1/openclaw  # OpenClaw persistent memory
    - /volume1/logs      # PII and system logs
```

---

## 16. Kubernetes Migration Roadmap

(Phased roadmap unchanged, summarized.)

- **Phase 1 (0–3 months)**: Learn k3s, migrate observability
- **Phase 2 (3–9 months)**: Coexist Docker Compose + Kubernetes
- **Phase 3 (9–18 months)**: Migrate core services
- **Phase 4 (18+ months)**: Full Kubernetes

---

## 17. Fine-Tuning Infrastructure

### 17.1 LoRA on DGX Spark #2 (Pseudocode)

```
# Pseudocode: LoRA fine-tuning configuration

function create_lora_config():
  lora_config = {
    "rank": 16,
    "alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
  }
  
  return lora_config


function apply_lora_to_model(base_model, lora_config):
  # Apply LoRA adapter to base model
  peft_model = get_peft_model(base_model, lora_config)
  
  return peft_model


# Example usage
config = create_lora_config()
model = apply_lora_to_model(base_llama_model, config)
```

---

## 18. Multi-Modal Capabilities

### 18.1 Speech-to-Text (Whisper)

```yaml
services:
  whisper:
    image: onerahmet/openai-whisper-asr-webservice:latest-gpu
    ports:
      - "9000:9000"
```

### 18.2 Text-to-Speech

**Local (Piper):**

```yaml
services:
  piper:
    image: rhasspy/wyoming-piper:latest
    ports:
      - "10200:10200"
```

(Cloud ElevenLabs usage as previously specified.)

---

## 19. Quick Reference

### 19.1 Ollama Commands

```bash
# Pull model
ollama pull llama3.2:70b
ollama pull deepseek-v3:236b-q4_K_M

# Run interactively
ollama run mistral-small:latest

# List models
ollama list
```

### 19.2 Container Commands

```bash
# Build and push
docker build -t ds423.tailnet.ts.net:5000/pais/my-service:v1 .
docker push ds423.tailnet.ts.net:5000/pais/my-service:v1
```

### 19.3 Service URLs

| Service | URL | Port |
|---------|-----|------|
| Ollama (Mac Studio) | `mac-studio.tailnet.ts.net` | 11434 |
| Ollama (K6) | `k6.tailnet.ts.net` | 11434 |
| OpenClaw | `k6.tailnet.ts.net` | 3000 |
| AnythingLLM | `mac-mini.tailnet.ts.net` | 3001 |
| n8n | `mac-mini.tailnet.ts.net` | 5678 |
| Grafana | `mac-mini.tailnet.ts.net` | 3002 |
| Qdrant | `ai-max-395.tailnet.ts.net` | 6333 |
| Nemotron PII | `ai-max-395.tailnet.ts.net` | 8888 |

### 19.4 Cloud Services

| Service | Use Case | API Endpoint |
|---------|----------|--------------|
| NVIDIA Brev | 405B+ models | `api.brev.dev` |
| Vertex AI | Enterprise | `aiplatform.googleapis.com` |
| Gemini API | Fast/cheap | `generativelanguage.googleapis.com` |
| ElevenLabs | Voice | `api.elevenlabs.io` |

### 19.5 Hardware Quick Reference

| Device | RAM | Role | Primary Service |
|--------|-----|------|-----------------|
| Mac Studio M3 Ultra | 256GB | Inference | Ollama 70B-236B |
| Mac mini M4 Pro | 48GB | Orchestration | LangFlow, n8n |
| GMKtec AI Max+ 395 | 128GB | Knowledge | Qdrant, Redis |
| GMKtec K6 | 64GB | OpenClaw | Personal AI |
| ThinkPad P14s | 96GB | Mobile | Portable AI |
| MacBook Pro M4 Pro | 48GB | Mobile | Executive |
| Dell Pro Max 16 Plus | 32GB | Mobile | GPU work |

---

## Conclusion

PAIS v1.3.2 represents a **local-first with intelligent cloud overflow and proactive AI assistance** framework for sovereign AI infrastructure. This version adds OpenClaw for 24/7 personal AI assistance and updates hardware specifications for enhanced capabilities.

**Key architectural decisions in v1.3.2**:

1. **Ollama as primary inference**: Simple, unified API across all hardware
2. **Mac Studio M3 Ultra (256GB)**: Now supports 236B models locally
3. **OpenClaw on GMKtec K6**: Dedicated 24/7 personal AI assistant with system integration
4. **Mac mini for orchestration**: Efficient control plane
5. **AI Max+ 395 for knowledge**: Memory-intensive embeddings and caching
6. **Three mobile workstations**: On-the-go AI capabilities
7. **Intelligent cloud overflow**: NVIDIA Brev, Vertex/Gemini, ElevenLabs when needed
8. **Nemotron PII detection**: AI-powered context-aware PII filtering before cloud

**This enables**:
- 95%+ inference on-device (privacy, zero marginal cost)
- 24/7 proactive AI assistance via OpenClaw
- Support for 236B models locally (with 256GB Mac Studio)
- Seamless cloud scaling for 405B+ models
- Complete data sovereignty with explicit cloud consent
- Production-grade security and observability

The framework is designed to grow with your needs—start with Ollama on a single Mac, add OpenClaw for proactive assistance, expand to multi-device orchestration, add cloud overflow as required.

---

*End of PAIS Framework v1.3.2*

**Document Revision History:**
- v1.0 (June 2025): Initial framework
- v1.1 (September 2025): Tailscale networking
- v1.2 (January 2026): Containerization, observability
- v1.3 (February 2026): Security, multi-tenancy, fine-tuning
- v1.3.1 (February 2026): Ollama primary, cloud overflow, removed ASCII diagram
- v1.3.2 (February 2026): OpenClaw integration, Nemotron PII detection (AI-powered, pseudocode), updated hardware (Mac Studio 256GB, K6 64GB), mobile workstations
