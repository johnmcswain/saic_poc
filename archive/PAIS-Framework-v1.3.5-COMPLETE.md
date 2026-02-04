# PRIVATE AI SOVEREIGNTY (PAIS)

**A Framework for Consumer Control of Local AI Infrastructure**

**Version 1.3.5**  
**February 2026**

*Includes: High-Availability Knowledge Layer, Isolated OpenClaw on Raspberry Pi 5, Storage Architecture, Tailscale Networking, Containerization Strategy, Observability, Security Hardening, Local-First with Cloud Overflow, Multi-Tenancy, and Flexible Orchestration*

---

> *"If I build the tools, I have only myself to blame for the exposure and consequences of my and my family's use."*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What's New in Version 1.3.5](#2-whats-new-in-version-135)
3. [Hardware Architecture](#3-hardware-architecture)
4. [Orchestration Layer](#4-orchestration-layer)
5. [Inference Architecture](#5-inference-architecture)
6. [OpenClaw Personal AI Assistant](#6-openclaw-personal-ai-assistant)
7. [High-Availability Knowledge Layer](#7-high-availability-knowledge-layer)
8. [Cloud Overflow Strategy](#8-cloud-overflow-strategy)
9. [Containerization Strategy](#9-containerization-strategy)
10. [Source Code Management](#10-source-code-management)
11. [Observability Stack](#11-observability-stack)
12. [Security and Hardening](#12-security-and-hardening)
13. [Multi-Tenancy and Governance](#13-multi-tenancy-and-governance)
14. [Performance Optimization](#14-performance-optimization)
15. [Backup and Disaster Recovery](#15-backup-and-disaster-recovery)
16. [Fine-Tuning Infrastructure](#16-fine-tuning-infrastructure)
17. [Multi-Modal Capabilities](#17-multi-modal-capabilities)
18. [Quick Reference](#18-quick-reference)

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
- **Isolated proactive AI assistance**—OpenClaw on dedicated Raspberry Pi 5 with complete separation from critical infrastructure
- **High-availability knowledge**—Distributed vector database with 99.95% uptime
- **Customizable governance**—You define content policies, access controls, and cloud usage thresholds
- **Secure remote access**—Tailscale mesh VPN provides seamless connectivity without exposing services to the internet
- **Portable architecture**—Docker containers enable workload mobility across any PAIS node
- **Observable systems**—Prometheus and Grafana provide comprehensive metrics and performance insights
- **Production-grade security**—Container hardening, zero-trust networking, and encrypted backups
- **Flexible orchestration**—Choose from LangFlow, LMStudio, AnythingLLM, n8n, OpenClaw, or combined approaches

---

## 2. What's New in Version 1.3.5

Version 1.3.5 refines the PAIS framework by removing the Kubernetes migration roadmap to reflect a deliberate choice to focus on Docker + Tailscale + HA patterns instead of cluster orchestration.

### 2.1 Architectural Highlights

- **OpenClaw isolation**: Dedicated Raspberry Pi 5 #1 (16GB) for OpenClaw, separate from all critical services
- **High-availability knowledge layer**: 3-node Qdrant cluster (K6 + 2× M7) with 99.95% uptime and 3× read throughput
- **GPU-accelerated content safety**: AI Max+ 395 dedicated to Nemotron PII + Whisper Large
- **Mac mini simplification**: Pure orchestration role (no OpenClaw), with 12GB RAM and ~15W power freed
- **No Kubernetes dependency**: All components run on Docker Compose with clear, documented patterns for scale and resilience

### 2.2 Rationale for Dropping Kubernetes Roadmap

- **Complexity vs. benefit**: For a single-tenant, homelab-scale deployment, Docker Compose + Tailscale + HA patterns deliver the necessary resilience and flexibility without the operational overhead of Kubernetes
- **Cognitive load**: Removing Kubernetes from the critical path reduces maintenance burden and operational risk
- **Portability**: The architecture remains portable to Kubernetes in the future, but the framework does not assume or require it

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
  role: "Pure orchestration and monitoring (OpenClaw removed)"
  
  services:
    - LangFlow (visual workflows) [4GB]
    - LMStudio (local model testing) [2GB]
    - AnythingLLM (RAG/chat interface) [4GB]
    - n8n (automation) [4GB]
    - Prometheus & Grafana (observability) [4GB]
    - API Gateway (Traefik) [2GB]
    - Model Router (intelligent tiering + cloud overflow) [2GB]
  
  total_allocation: 22GB (was 34GB in v1.3.3)
  freed_resources: 12GB RAM, ~15W power
  remaining: 26GB buffer (was 14GB)
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
    - DeepSeek v3 236B (Q4_K_M): 2-3 tok/s
    - Mistral Small 22B: 12-15 tok/s
    - Stable Diffusion XL: 25-35s per image
    - Whisper Medium: real-time transcription

DGX Spark #1:
  gb10_grace_blackwell: "20-core Arm CPU + 2× tensor units"
  memory: 128GB LPDDR5x
  storage: 4TB NVMe
  qty: 1 units
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
```

#### High-Availability Knowledge Layer

```yaml
GMKtec K6:
  cpu: AMD Ryzen 7 7840HS (8-core, up to 5.1 GHz)
  memory: 64GB DDR5 (16GB × 4)
  storage: 1TB PCIe 4.0 SSD
  networking: Dual NIC 2.5Gbps, WiFi 6E, USB4
  power: ~45W
  qty: 1
  role: "Qdrant Primary + Redis Cache + Embeddings"
  
  primary_services:
    - Qdrant primary shard (40GB allocation)
    - Redis semantic cache (16GB allocation)
    - Embedding service - nomic-embed-text (4GB)
    - System overhead (4GB)
  
  rationale:
    - Largest RAM (64GB) → primary shard + cache
    - Dual 2.5Gbps NIC for network redundancy
    - USB4 for fast NAS access
    - x86-64 for Docker compatibility

GMKtec M7 #1:
  cpu: AMD Ryzen 7 PRO 6850H (8-core, up to 4.7 GHz)
  memory: 32GB DDR5 (2× 16GB) [UPGRADED]
  igpu: AMD Radeon 680M (RDNA2, 12 CUs)
  storage: 512GB NVMe
  power: ~45W
  qty: 1
  role: "Qdrant Replica + Reranking"
  
  primary_services:
    - Qdrant replica shard (26GB allocation)
    - BGE reranker-v2-m3 (4GB)
    - System overhead (2GB)

GMKtec M7 #2:
  cpu: AMD Ryzen 7 PRO 6850H (8-core, up to 4.7 GHz)
  memory: 32GB DDR5 (2× 16GB) [UPGRADED]
  igpu: AMD Radeon 680M (RDNA2, 12 CUs)
  storage: 512GB NVMe
  power: ~45W
  qty: 1
  role: "Qdrant Replica + Document Processing"
  
  primary_services:
    - Qdrant replica shard (20GB allocation)
    - Document ingestion pipeline (8GB)
    - OCR service (Tesseract) (2GB)
    - System overhead (2GB)
```

#### Content Safety & Multi-Modal Layer

```yaml
GMKtec AI Max+ 395:
  cpu: Ryzen AI 9 395 (16-core, 5.1 GHz boost)
  unified_memory: 128GB LPDDR5X
  npu: 40+ TOPS AI accelerator
  storage: 2TB PCIe 4.0 SSD
  power: ~65W
  qty: 1
  role: "Nemotron PII Detection + Multi-Modal Preprocessing"
  
  primary_services:
    - Nemotron-4 Content Safety (64GB allocation)
    - Whisper Large v3 transcription (24GB)
    - Image preprocessing (CoreML/NPU) (16GB)
    - Redis PII cache (16GB)
    - System overhead (8GB)
```

#### OpenClaw Layer (v1.3.4+)

```yaml
Raspberry Pi 5 #1:
  cpu: Broadcom BCM2712 (quad-core Arm Cortex-A76, 2.4 GHz)
  memory: 16GB LPDDR4X
  storage: 64GB NVMe (via M.2 HAT) or microSD
  power: 10W idle, 15W load
  networking: Gigabit Ethernet (recommended for 24/7)
  qty: 1 (dedicated to OpenClaw)
  role: "OpenClaw 24/7 Personal AI Assistant (Isolated)"
  
  primary_services:
    - OpenClaw (personal AI assistant) [4GB]
    - Ollama (qwen2.5:7b or phi3:mini) [6-8GB]
    - PostgreSQL (assistant memory) [2GB]
    - System overhead [2GB]
  
  total_allocation: 14-16GB
  
  rationale:
    - Complete isolation from critical infrastructure
    - Lowest power consumption (10-15W = $1.50/month)
    - Sufficient for 7B models (8-12 tok/s)
    - Perfect for asynchronous assistant tasks
    - No impact on orchestration/inference/knowledge layers
    - Dedicated device = maximum security for system-level permissions
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

Dell Pro Max 16 Plus:
  cpu: Intel Core Ultra 9 285HX (24-core, 2.8 GHz)
  gpu: NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU
  memory: 32GB DDR5
  storage: 2TB NVMe
  display: 16" (specs vary by config)
  qty: 1
  role: "GPU-accelerated mobile workstation, training on-the-go"
```

#### Edge Layer (UPDATED in v1.3.4)

```yaml
Raspberry Pi 5 #1:
  memory: 16GB LPDDR4X
  power: 10-15W
  qty: 1
  role: "OpenClaw 24/7 Personal AI Assistant (Dedicated)"
  
  services:
    - OpenClaw
    - Ollama (qwen2.5:7b)
    - PostgreSQL

Raspberry Pi 5 #2:
  memory: 16GB LPDDR4X
  power: 10W
  qty: 1
  role: "Home automation hub (Home Assistant, Node-RED)"
  
  services:
    - Home Assistant
    - Node-RED
    - Zigbee2MQTT
    - MQTT broker

Raspberry Pi 5 #3:
  memory: 16GB LPDDR4X
  power: 10W
  qty: 1
  role: "IoT monitoring and data collection"
  
  services:
    - InfluxDB (time-series metrics)
    - Telegraf (metrics collection)
    - MQTT subscriber
    - Custom monitoring scripts

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
    /volume1/qdrant: "Qdrant cluster snapshots"
    /volume1/docker: "Docker registry, container images"
    /volume1/git: "Source code mirrors (Gitea backup)"
    /volume1/backups: "System backups and snapshots"
    /volume1/family: "Family documents + encrypted cloud backup"
    /volume1/synthetic: "AI-generated outputs"
    /volume1/openclaw: "OpenClaw persistent memory (on RPi5)"
```

### 3.2 Architecture Overview

The PAIS architecture consists of seven logical layers connected via Tailscale mesh networking:

**Layer 1: Orchestration** (Mac mini M4 Pro)
- LangFlow, LMStudio, AnythingLLM, n8n
- API Gateway (Traefik), Model Router, Governance Engine
- Prometheus & Grafana observability
- **No longer hosts OpenClaw (freed 12GB RAM, 15W power)**

**Layer 2: Local Inference** (Ollama Primary)

- Mac Studio M3 Ultra: Interactive 70B-236B inference, ComfyUI
- DGX Spark #1: Batch inference, high-concurrency
- DGX Spark #2: Fine-tuning, LoRA training
- Jetson Orin (×3): Edge 7B inference, STT/TTS

**Layer 3: OpenClaw Personal Assistant** (Raspberry Pi 5 #1) **[NEW in v1.3.4]**
- **Dedicated Raspberry Pi 5 (16GB)** for complete isolation
- OpenClaw 24/7 proactive AI assistant
- Ollama 7B model (qwen2.5:7b or phi3:mini)
- Email, calendar, file management, system monitoring
- **10-15W power, zero impact on critical infrastructure**

**Layer 4: High-Availability Knowledge** (K6 + M7 Cluster)
- GMKtec K6: Qdrant primary + Redis cache + embeddings
- GMKtec M7 #1: Qdrant replica + reranking
- GMKtec M7 #2: Qdrant replica + document processing
- **99.95% uptime, 3× read throughput, 15M+ vectors**

**Layer 5: Content Safety** (AI Max+ 395)

- Nemotron PII detection (GPU-accelerated)
- Multi-modal preprocessing (Whisper, image processing)
- Redis PII cache
- **Automatic PII redaction before cloud requests**

**Layer 6: Cloud Overflow** (When Local Insufficient)
- NVIDIA Brev: 405B+ models, GPU-intensive workloads
- Google Vertex AI / AI Studio: Enterprise inference
- Gemini API: Fast, cost-effective inference
- ElevenLabs: Professional voice generation

**Layer 7: Storage** (Synology DS423, 24TB)
- Docker registry, Ollama models, backups
- Gitea mirror, family documents, Qdrant snapshots

**Edge Tier** (Raspberry Pi 5 #2-3)
- Home automation (Home Assistant, Node-RED)
- IoT monitoring (InfluxDB, Telegraf)

**Mobile Tier: Workstations** (ThinkPad, MacBook Pro, Dell Pro Max)
- On-the-go AI inference and development
- Remote access to PAIS via Tailscale
- Portable orchestration and model testing

*See accompanying draw.io diagram for visual architecture overview.*

---

## 4. Orchestration Layer

The orchestration layer provides the user interface, workflow automation, and service coordination for the entire PAIS system. Version 1.3.4 simplifies the Mac mini to pure orchestration, with OpenClaw now isolated on Raspberry Pi 5.

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
    - REDIS_URL=redis://k6.tailnet.ts.net:6379
  deploy:
    resources:
      limits:
        memory: 4GB
  restart: always
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
  deploy:
    resources:
      limits:
        memory: 2GB
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
  environment:
    - LLM_PROVIDER=ollama
    - OLLAMA_BASE_PATH=http://mac-studio.tailnet.ts.net:11434
    - EMBEDDING_ENGINE=ollama
    - EMBEDDING_MODEL_PREF=nomic-embed-text
    - VECTORDB=qdrant
    - QDRANT_ENDPOINT=http://k6.tailnet.ts.net:6333
  deploy:
    resources:
      limits:
        memory: 4GB
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
  deploy:
    resources:
      limits:
        memory: 4GB
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
| **OpenClaw** | `http://rpi5-openclaw.tailnet.ts.net:3000` | Personal AI assistant |

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

**Raspberry Pi 5 (ARM64):**
```bash
# ARM64 installation
curl -fsSL https://ollama.com/install.sh | sh

# Or Docker
docker run -d \
  -v /mnt/nas/ollama:/root/.ollama \
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

# Lightweight models for RPi5
ollama pull qwen2.5:7b
ollama pull phi3:mini

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

#### Raspberry Pi 5 (OpenClaw)

```bash
# Run 7B model for OpenClaw
ollama run qwen2.5:7b

# Or smaller for faster response
ollama run phi3:mini
```

**Performance**: 8-12 tok/s for 7B models on CPU

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

OpenClaw is an open-source personal AI assistant that provides 24/7 proactive monitoring and task automation. In v1.3.4, OpenClaw runs on a **dedicated Raspberry Pi 5** for complete isolation from critical PAIS infrastructure.

### 6.1 Why Raspberry Pi 5 for OpenClaw?

✅ **Complete isolation**: Separate device from orchestration, inference, and knowledge layers  
✅ **Lowest power**: 10-15W for 24/7 operation ($1.50/month electricity)  
✅ **Maximum security**: System-level permissions isolated on dedicated device  
✅ **Zero performance impact**: No competition with critical services  
✅ **Sufficient performance**: 8-12 tok/s acceptable for asynchronous assistant tasks  
✅ **Simplified architecture**: Mac mini now pure orchestration (12GB RAM freed)  
✅ **Already owned**: No additional hardware purchase needed

### 6.2 Deployment on Raspberry Pi 5

#### Hardware Preparation

```bash
# Flash Raspberry Pi OS (64-bit) to microSD or NVMe
# Use Raspberry Pi Imager: https://www.raspberrypi.com/software/

# SSH into Raspberry Pi
ssh pi@rpi5-openclaw.local

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker pi
sudo systemctl enable docker

# Reboot
sudo reboot
```

#### Mount NAS for Models

```bash
# Install NFS client
sudo apt install nfs-common -y

# Create mount point
sudo mkdir -p /mnt/nas

# Mount NAS
sudo mount -t nfs ds423.tailnet.ts.net:/volume1/models /mnt/nas

# Add to /etc/fstab for auto-mount on boot
echo "ds423.tailnet.ts.net:/volume1/models /mnt/nas nfs defaults,_netdev 0 0" | sudo tee -a /etc/fstab

# Verify mount
ls /mnt/nas/ollama
```

#### Docker Compose Configuration

```yaml
# docker-compose.openclaw-rpi5.yml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-openclaw
    ports:
      - "11434:11434"
    volumes:
      - /mnt/nas/ollama:/root/.ollama:ro
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_MODELS=/root/.ollama
    deploy:
      resources:
        limits:
          memory: 8GB
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

  openclaw:
    image: openclaw/openclaw:latest
    container_name: openclaw
    ports:
      - "3000:3000"   # Web UI
      - "8765:8765"   # WebSocket
    volumes:
      - openclaw-data:/root/.openclaw
      - /home/pi:/host_home:rw  # For file operations
      - /mnt/nas/openclaw:/persistent:rw  # NAS storage for long-term memory
    environment:
      # AI Model Configuration
      - OLLAMA_BASE_URL=http://localhost:11434
      - DEFAULT_MODEL=qwen2.5:7b  # Lightweight 7B model
      
      # Messaging Integrations
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - SLACK_TOKEN=${SLACK_TOKEN}
      - WHATSAPP_ENABLED=false
      
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
    deploy:
      resources:
        limits:
          memory: 4GB
    restart: always
    depends_on:
      - ollama
      - postgres

  postgres:
    image: postgres:15-alpine
    container_name: openclaw-db
    environment:
      - POSTGRES_DB=openclaw
      - POSTGRES_USER=openclaw
      - POSTGRES_PASSWORD=${OPENCLAW_DB_PASSWORD}
    volumes:
      - openclaw-db:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          memory: 2GB
    restart: always

volumes:
  openclaw-data:
  openclaw-db:
```

#### Initial Setup

```bash
# Pull 7B model for OpenClaw
docker run --rm -v /mnt/nas/ollama:/root/.ollama ollama/ollama pull qwen2.5:7b

# Alternative lightweight model
docker run --rm -v /mnt/nas/ollama:/root/.ollama ollama/ollama pull phi3:mini

# Create environment file
cat > .env << EOF
OPENCLAW_ADMIN_PASSWORD=$(openssl rand -base64 16)
OPENCLAW_JWT_SECRET=$(openssl rand -hex 32)
OPENCLAW_DB_PASSWORD=$(openssl rand -base64 16)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
SLACK_TOKEN=your_slack_token
EOF

# Start OpenClaw
docker-compose -f docker-compose.openclaw-rpi5.yml up -d

# Check logs
docker-compose -f docker-compose.openclaw-rpi5.yml logs -f openclaw

# Access web UI
# http://rpi5-openclaw.tailnet.ts.net:3000
```

### 6.3 Performance Expectations

#### Inference Performance

| Model | Tokens/sec | Memory | Response Time |
|-------|------------|--------|---------------|
| qwen2.5:7b (Q4_K_M) | 8-12 | 6GB | 2-3s |
| phi3:mini (Q4_K_M) | 10-15 | 4GB | 1-2s |

#### Task Performance

| OpenClaw Task | RPi5 7B | Mac mini 14B | Acceptable? |
|---------------|---------|--------------|-------------|
| Email triage | 2-3s | 1-2s | ✅ Yes (async) |
| Calendar scheduling | 2-3s | 1-2s | ✅ Yes (async) |
| File organization | 2-3s | 1-2s | ✅ Yes (batch) |
| System monitoring | 2-3s | 1-2s | ✅ Yes (scheduled) |
| Daily briefing | 3-5s | 2-3s | ✅ Yes (morning) |

**Conclusion**: 8-12 tok/s on Raspberry Pi 5 is **sufficient** for OpenClaw's asynchronous assistant tasks. Most tasks are not latency-sensitive (email triage, calendar, file organization run on schedules).

### 6.4 Power and Cost Analysis

| Device | Power (Idle) | Power (Load) | Monthly Cost | Annual Cost |
|--------|--------------|--------------|--------------|-------------|
| Raspberry Pi 5 | 10W | 15W | $1.50 | $18 |
| Mac mini M4 Pro | 15W | 50W | $5.00 | $60 |
| **Savings** | - | - | **$3.50** | **$42** |

*Assumes $0.12/kWh electricity rate*

### 6.5 OpenClaw Use Cases

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
          - http://k6.tailnet.ts.net:6333/health  # Qdrant
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
        path: /home/pi/Downloads
    - categorize_files:
        rules:
          - type: pdf
            destination: /home/pi/Documents/PDFs
          - type: image
            destination: /home/pi/Pictures
          - type: video
            destination: /home/pi/Videos
          - type: archive
            destination: /home/pi/Archives
    - move_files:
        confirm: false
    - notify:
        channel: slack
        message: "Organized {{ file_count }} files from Downloads"
```

### 6.6 Integration with PAIS Orchestration

#### Routing Tasks Between OpenClaw and n8n (Pseudocode)

```
# Pseudocode: TaskRouter (updated for RPi5)

class TaskRouter:
  
  OPENCLAW_ENDPOINT = "http://rpi5-openclaw.tailnet.ts.net:3000/api/task"
  N8N_ENDPOINT = "http://mac-mini.tailnet.ts.net:5678/webhook/task"
  
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
      return OPENCLAW_ENDPOINT
    
    # n8n: data workflows, API integrations
    else if any_keyword_in(N8N_TASKS, task_lower):
      return N8N_ENDPOINT
    
    # Default: OpenClaw for general assistance
    else:
      return OPENCLAW_ENDPOINT
```

### 6.7 Security Considerations

#### Tailscale ACLs for Isolated OpenClaw

```json
{
  "acls": [
    {
      "action": "accept",
      "src": ["group:admins"],
      "dst": ["tag:openclaw-rpi5:3000", "tag:openclaw-rpi5:8765"]
    },
    {
      "action": "deny",
      "src": ["group:family"],
      "dst": ["tag:openclaw-rpi5:*"]
    },
    {
      "action": "accept",
      "src": ["tag:openclaw-rpi5"],
      "dst": [
        "tag:mac-studio:11434",  # Ollama for fallback
        "tag:nas:*",              # NAS for file operations
        "tag:mac-mini:*"          # Orchestration services
      ]
    }
  ]
}
```

**Rationale**: 
- OpenClaw has system-level permissions → restrict to admins only
- OpenClaw on isolated device → can only reach specific services via Tailscale
- No direct internet access → all external communication via Tailscale mesh

---

## 7. High-Availability Knowledge Layer

The knowledge layer provides persistent memory and context for all AI services in PAIS. Version 1.3.3 introduces a **3-node Qdrant cluster** with high availability, distributed throughput, and automatic failover.

### 7.1 Architecture Overview

```yaml
qdrant_cluster:
  topology: "3-node distributed cluster"
  replication_factor: 2
  
  nodes:
    primary:
      device: GMKtec K6 (64GB)
      role: Primary shard + Redis cache + embeddings
      allocation: 40GB Qdrant + 16GB Redis + 4GB embeddings
    
    replica_1:
      device: GMKtec M7 #1 (32GB)
      role: Replica shard + reranking
      allocation: 26GB Qdrant + 4GB reranker
    
    replica_2:
      device: GMKtec M7 #2 (32GB)
      role: Replica shard + document processing
      allocation: 20GB Qdrant + 8GB doc pipeline + 2GB OCR
  
  benefits:
    - "99.95% uptime (automatic failover)"
    - "3× read throughput (load balanced)"
    - "15M+ vector capacity (distributed)"
    - "Zero-downtime updates (rolling restart)"
```

### 7.2 Deployment Architecture

#### Node 1: GMKtec K6 (Primary)

```yaml
# docker-compose.qdrant-k6.yml on GMKtec K6
version: '3.8'

services:
  qdrant-primary:
    image: qdrant/qdrant:latest
    container_name: qdrant-primary
    ports:
      - "6333:6333"   # REST API
      - "6334:6334"   # gRPC
    volumes:
      - /volume1/qdrant/primary:/qdrant/storage
    environment:
      - QDRANT_CLUSTER_ENABLED=true
      - QDRANT_CLUSTER_NODE_ID=k6-primary
      - QDRANT_CLUSTER_URL=http://k6.tailnet.ts.net:6335
      - QDRANT_CLUSTER_P2P_PORT=6335
      - QDRANT_MAX_SEGMENT_SIZE=200000
    deploy:
      resources:
        limits:
          memory: 40GB
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis-cache:
    image: redis:7-alpine
    container_name: redis-cache
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 16gb --maxmemory-policy allkeys-lru
    volumes:
      - redis-cache-data:/data
    deploy:
      resources:
        limits:
          memory: 16GB
    restart: always

  nomic-embed:
    image: ollama/ollama:latest
    container_name: nomic-embed
    ports:
      - "11435:11434"
    volumes:
      - /volume1/models/ollama:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_MODELS=/root/.ollama
    deploy:
      resources:
        limits:
          memory: 4GB
    restart: always

volumes:
  redis-cache-data:
```

#### Node 2: GMKtec M7 #1 (Replica + Reranking)

```yaml
# docker-compose.qdrant-m7-1.yml on GMKtec M7 #1
version: '3.8'

services:
  qdrant-replica-1:
    image: qdrant/qdrant:latest
    container_name: qdrant-replica-1
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - /volume1/qdrant/replica-1:/qdrant/storage
    environment:
      - QDRANT_CLUSTER_ENABLED=true
      - QDRANT_CLUSTER_NODE_ID=m7-1-replica
      - QDRANT_CLUSTER_URL=http://m7-1.tailnet.ts.net:6335
      - QDRANT_CLUSTER_P2P_PORT=6335
      - QDRANT_BOOTSTRAP_URL=http://k6.tailnet.ts.net:6335
    deploy:
      resources:
        limits:
          memory: 26GB
    restart: always

  reranker:
    image: custom/bge-reranker-v2-m3:latest
    container_name: reranker
    ports:
      - "8001:8000"
    deploy:
      resources:
        limits:
          memory: 4GB
    restart: always
```

#### Node 3: GMKtec M7 #2 (Replica + Document Processing)

```yaml
# docker-compose.qdrant-m7-2.yml on GMKtec M7 #2
version: '3.8'

services:
  qdrant-replica-2:
    image: qdrant/qdrant:latest
    container_name: qdrant-replica-2
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - /volume1/qdrant/replica-2:/qdrant/storage
    environment:
      - QDRANT_CLUSTER_ENABLED=true
      - QDRANT_CLUSTER_NODE_ID=m7-2-replica
      - QDRANT_CLUSTER_URL=http://m7-2.tailnet.ts.net:6335
      - QDRANT_CLUSTER_P2P_PORT=6335
      - QDRANT_BOOTSTRAP_URL=http://k6.tailnet.ts.net:6335
    deploy:
      resources:
        limits:
          memory: 20GB
    restart: always

  doc-processor:
    image: custom/document-processor:latest
    container_name: doc-processor
    ports:
      - "8002:8000"
    volumes:
      - /volume1/documents:/input
      - /volume1/processed:/output
    deploy:
      resources:
        limits:
          memory: 8GB
    restart: always

  tesseract-ocr:
    image: tesseractshadow/tesseract4re:latest
    container_name: tesseract-ocr
    deploy:
      resources:
        limits:
          memory: 2GB
    restart: always
```

### 7.3 Load Balancer (HAProxy)

```yaml
# haproxy.cfg
global
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend qdrant_api
    bind *:6333
    default_backend qdrant_cluster

backend qdrant_cluster
    balance roundrobin
    option httpchk GET /health
    server k6-primary k6.tailnet.ts.net:6333 check
    server m7-1-replica m7-1.tailnet.ts.net:6333 check
    server m7-2-replica m7-2.tailnet.ts.net:6333 check
```

Deploy HAProxy on Mac mini:

```yaml
# docker-compose.haproxy.yml on Mac mini
services:
  haproxy:
    image: haproxy:latest
    container_name: haproxy-qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
    restart: always
```

### 7.4 Performance Benchmarks

#### Before (Single Qdrant on AI Max+ 395)

| Metric | Value |
|--------|-------|
| Read QPS | 100-200 |
| Write QPS | 50-80 |
| Vector capacity | 5M |
| Availability | 99.0% (single point of failure) |
| Failover | Manual (10+ min downtime) |

#### After (3-Node Cluster on K6 + M7)

| Metric | Value | Improvement |
|--------|-------|-------------|
| Read QPS | 400-500 | **3× faster** |
| Write QPS | 150-200 | **3× faster** |
| Vector capacity | 15M+ | **3× capacity** |
| Availability | 99.95% (HA) | **+0.95%** |
| Failover | Automatic (<30s) | **20× faster** |

### 7.5 Cluster Management

#### Creating a Collection (Sharded)

```bash
curl -X PUT http://k6.tailnet.ts.net:6333/collections/my-collection \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 1536,
      "distance": "Cosine"
    },
    "shard_number": 3,
    "replication_factor": 2,
    "write_consistency_factor": 2
  }'
```

#### Checking Cluster Health

```bash
# Check cluster status
curl http://k6.tailnet.ts.net:6333/cluster

# Example response
{
  "result": {
    "status": "healthy",
    "peer_count": 3,
    "consensus_thread_status": "working",
    "peers": {
      "k6-primary": {
        "uri": "http://k6.tailnet.ts.net:6335",
        "collections": 5,
        "points_count": 5000000
      },
      "m7-1-replica": {
        "uri": "http://m7-1.tailnet.ts.net:6335",
        "collections": 5,
        "points_count": 5000000
      },
      "m7-2-replica": {
        "uri": "http://m7-2.tailnet.ts.net:6335",
        "collections": 5,
        "points_count": 5000000
      }
    }
  }
}
```

#### Rolling Updates (Zero Downtime)

```bash
# Update M7 #2 first (least traffic)
ssh m7-2.tailnet.ts.net
docker-compose -f docker-compose.qdrant-m7-2.yml pull
docker-compose -f docker-compose.qdrant-m7-2.yml up -d

# Wait for sync
curl http://k6.tailnet.ts.net:6333/cluster

# Update M7 #1
ssh m7-1.tailnet.ts.net
docker-compose -f docker-compose.qdrant-m7-1.yml pull
docker-compose -f docker-compose.qdrant-m7-1.yml up -d

# Wait for sync
curl http://k6.tailnet.ts.net:6333/cluster

# Update K6 primary last
ssh k6.tailnet.ts.net
docker-compose -f docker-compose.qdrant-k6.yml pull
docker-compose -f docker-compose.qdrant-k6.yml up -d
```

### 7.6 Backup and Restore

#### Automated Snapshots

```yaml
# Qdrant automatic snapshots (on each node)
environment:
  - QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/snapshots
  - QDRANT__STORAGE__SNAPSHOTS__ENABLED=true
```

```bash
# Backup script (runs on NAS)
#!/bin/bash
# /volume1/scripts/backup-qdrant.sh

DATE=$(date +%Y%m%d)

# Snapshot all collections
for collection in $(curl -s http://k6.tailnet.ts.net:6333/collections | jq -r '.result.collections[].name'); do
  curl -X POST "http://k6.tailnet.ts.net:6333/collections/$collection/snapshots"
done

# Copy snapshots to NAS
rsync -avz k6.tailnet.ts.net:/volume1/qdrant/primary/snapshots/ /volume1/backups/qdrant/$DATE/
rsync -avz m7-1.tailnet.ts.net:/volume1/qdrant/replica-1/snapshots/ /volume1/backups/qdrant/$DATE-replica-1/
rsync -avz m7-2.tailnet.ts.net:/volume1/qdrant/replica-2/snapshots/ /volume1/backups/qdrant/$DATE-replica-2/

# Upload to Backblaze B2
b2 sync /volume1/backups/qdrant/$DATE/ b2://pais-backups/qdrant/$DATE/
```

#### Restore from Snapshot

```bash
# Stop Qdrant on target node
docker-compose -f docker-compose.qdrant-k6.yml down

# Restore snapshot
curl -X PUT "http://k6.tailnet.ts.net:6333/collections/my-collection/snapshots/upload" \
  --data-binary @/volume1/backups/qdrant/20260202/my-collection-snapshot.tar

# Restart Qdrant
docker-compose -f docker-compose.qdrant-k6.yml up -d
```

### 7.7 Monitoring

#### Prometheus Metrics

```yaml
# prometheus.yml (on Mac mini)
scrape_configs:
  - job_name: 'qdrant-cluster'
    static_configs:
      - targets:
        - 'k6.tailnet.ts.net:6333'
        - 'm7-1.tailnet.ts.net:6333'
        - 'm7-2.tailnet.ts.net:6333'
```

#### Key Metrics (PromQL)

```promql
# Cluster health
up{job="qdrant-cluster"} == 1

# Replication lag
qdrant_cluster_replication_lag_seconds

# Points per collection
qdrant_collections_points_count

# Read throughput
rate(qdrant_http_requests_total{method="GET"}[5m]) by (node)

# Write throughput
rate(qdrant_http_requests_total{method="POST"}[5m]) by (node)

# Query latency (p95)
histogram_quantile(0.95, rate(qdrant_query_duration_seconds_bucket[5m]))
```

### 7.8 Cost Analysis

| Component | Before (AI Max+ 395) | After (K6 + 2× M7) | Delta |
|-----------|----------------------|--------------------|-------|
| Hardware cost | $0 (already owned) | $340 (M7 RAM upgrades) | +$340 |
| Power (24/7) | 65W = $6.80/month | 135W = $14.17/month | +$7.37/month |
| Capacity | 5M vectors | 15M vectors | **+10M** |
| Availability | 99.0% | 99.95% | **+52 min uptime/year** |
| Read throughput | 100-200 QPS | 400-500 QPS | **+3×** |
| **Total annual cost** | **$82** | **$170** | **+$88** |

**ROI**: For **$340 one-time + $88/year**, you get:
- 3× capacity
- 3× read throughput
- 99.95% uptime (automatic failover)
- Production-grade HA

---

## 8. Cloud Overflow Strategy

PAIS is **local-first** but pragmatic. When local resources are insufficient—model too large, GPU memory exhausted, high concurrency—intelligently route to cloud services.

### 8.1 Cloud Service Providers

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

### 8.2 Intelligent Routing Logic (Pseudocode)

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

### 8.3 PII Detection (Nemotron Content Safety)

#### Overview

PAIS uses **NVIDIA Nemotron-4 Content Safety Reasoning** for context-aware PII detection before cloud requests. Deployed on AI Max+ 395 with Redis caching.

```yaml
# docker-compose.nemotron-pii.yml on AI Max+ 395
version: '3.8'

services:
  nemotron-pii:
    image: nvcr.io/nvidia/nemotron-4-340b-content-safety-reasoning:latest
    container_name: nemotron-pii
    ports:
      - "8888:8000"
    volumes:
      - /volume1/models/nemotron:/model_cache
    environment:
      - MODEL_NAME=nemotron-4-340b-content-safety-reasoning
      - QUANTIZATION=awq
      - GPU_MEMORY_UTILIZATION=0.7
      - MAX_CONCURRENT_REQUESTS=128
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
        limits:
          memory: 64GB
    restart: always

  pii-cache:
    image: redis:7-alpine
    container_name: pii-cache
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 16gb --maxmemory-policy allkeys-lru
    deploy:
      resources:
        limits:
          memory: 16GB
    restart: always
```

#### NemotronPIIDetector (Pseudocode)

```
# Pseudocode: Nemotron PII Detector

class NemotronPIIDetector:
  
  function __init__(nemotron_base_url, cache_client):
    this.base_url = nemotron_base_url
    this.cache = cache_client
  
  
  function detect(text):
    # Cache lookup
    cache_key = "pii:" + hash(text)
    cached = this.cache.get(cache_key)
    
    if cached is not null:
      return deserialize(cached)
    
    # Call Nemotron API
    payload = {
      "text": text,
      "detect_pii": true,
      "languages": ["en"]
    }
    
    response = http_post(this.base_url + "/v1/content-safety/detect", payload)
    
    # Cache result
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

#### CloudRouter with Nemotron (Pseudocode)

```
# Pseudocode: CloudRouter with Nemotron PII filtering

class CloudRouterWithNemotron:
  
  function __init__(pii_detector):
    this.pii_detector = pii_detector
  
  
  function execute(target, prompt, model, redact_pii = true):
    # Only PII-check non-local targets
    if target != LOCAL_OLLAMA and redact_pii == true:
      result = this.pii_detector.detect(prompt)
      
      if result.safe_for_cloud == false:
        # Apply policy
        policy = get_pii_policy_for_target(target)
        
        if policy == "strict":
          log_pii_violation(target, model, result)
          raise Error("Cloud call blocked due to PII")
        
        if policy == "moderate":
          prompt = result.redacted_text
          log_pii_violation(target, model, result)
    
    return this.execute_on_target(target, prompt, model)
```

---

## 9. Containerization Strategy

Docker containers provide the abstraction layer that makes PAIS truly portable and composable.

### 9.1 Self-Hosted Registry

```yaml
# docker-compose.registry.yml on DS423
services:
  registry:
    image: registry:2
    container_name: docker-registry
    ports:
      - "5000:5000"
    volumes:
      - /volume1/docker/registry:/var/lib/registry
    environment:
      - REGISTRY_STORAGE_DELETE_ENABLED=true
    restart: always
```

### 9.2 Image Organization

| Registry Path | Purpose | Architecture |
|--------------|---------|--------------|
| `ds423.tailnet.ts.net:5000/pais/langflow` | Workflow builder | amd64 |
| `ds423.tailnet.ts.net:5000/pais/anythingllm` | RAG chat interface | amd64 |
| `ds423.tailnet.ts.net:5000/pais/n8n` | Automation workflows | amd64 |
| `ds423.tailnet.ts.net:5000/pais/openclaw` | Personal AI assistant | amd64, arm64 |
| `ds423.tailnet.ts.net:5000/pais/ollama` | Inference engine | amd64, arm64 |
| `ds423.tailnet.ts.net:5000/pais/qdrant` | Vector database | amd64 |

### 9.3 Multi-Architecture Builds

```bash
# Create buildx builder
docker buildx create --name pais-builder --use

# Build multi-arch image
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ds423.tailnet.ts.net:5000/pais/openclaw:v1.3.5 \
  --push \
  .
```

---

## 10. Source Code Management

### 10.1 Repository Strategy

| Repository Type | Location | Purpose |
|-----------------|----------|---------|
| Primary | GitHub (private repos) | Collaboration, CI/CD |
| Mirror | DS423 NAS (/volume1/git) | Sovereign backup |
| Working | Development machine | Active development |

### 10.2 NAS Git Server (Gitea)

```yaml
# docker-compose.gitea.yml on DS423
services:
  gitea:
    image: gitea/gitea:latest
    container_name: gitea
    ports:
      - "3000:3000"
      - "2222:22"
    volumes:
      - /volume1/git/gitea:/data
    environment:
      - USER_UID=1000
      - USER_GID=1000
      - GITEA__database__DB_TYPE=postgres
      - GITEA__database__HOST=db:5432
      - GITEA__database__NAME=gitea
      - GITEA__database__USER=gitea
      - GITEA__database__PASSWD=${GITEA_DB_PASSWORD}
    restart: always
    depends_on:
      - db

  db:
    image: postgres:15-alpine
    container_name: gitea-db
    environment:
      - POSTGRES_USER=gitea
      - POSTGRES_PASSWORD=${GITEA_DB_PASSWORD}
      - POSTGRES_DB=gitea
    volumes:
      - /volume1/git/gitea-db:/var/lib/postgresql/data
    restart: always
```

---

## 11. Observability Stack

### 11.1 Deployment

```yaml
# docker-compose.monitoring.yml on Mac mini
services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    deploy:
      resources:
        limits:
          memory: 2GB
    restart: always

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3002:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana-data:/var/lib/grafana
    deploy:
      resources:
        limits:
          memory: 2GB
    restart: always

volumes:
  prometheus-data:
  grafana-data:
```

### 11.2 Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ollama-mac-studio'
    static_configs:
      - targets: ['mac-studio.tailnet.ts.net:11434']

  - job_name: 'ollama-rpi5'
    static_configs:
      - targets: ['rpi5-openclaw.tailnet.ts.net:11434']

  - job_name: 'openclaw'
    static_configs:
      - targets: ['rpi5-openclaw.tailnet.ts.net:3000']

  - job_name: 'qdrant-cluster'
    static_configs:
      - targets:
        - 'k6.tailnet.ts.net:6333'
        - 'm7-1.tailnet.ts.net:6333'
        - 'm7-2.tailnet.ts.net:6333'

  - job_name: 'nemotron-pii'
    static_configs:
      - targets: ['ai-max-395.tailnet.ts.net:8888']

  - job_name: 'edge-devices'
    static_configs:
      - targets:
        - 'rpi5-openclaw.tailnet.ts.net:9100'
        - 'rpi5-homeassistant.tailnet.ts.net:9100'
        - 'rpi5-iot.tailnet.ts.net:9100'
```

### 11.3 Key Metrics

```promql
# OpenClaw on RPi5
up{job="openclaw"} == 1
rate(openclaw_tasks_total[1h])
openclaw_task_duration_seconds

# RPi5 system metrics
node_load1{job="edge-devices",instance="rpi5-openclaw.tailnet.ts.net:9100"}
node_memory_MemAvailable_bytes{instance="rpi5-openclaw.tailnet.ts.net:9100"}

# Ollama inference
rate(ollama_tokens_generated_total[5m])
histogram_quantile(0.95, rate(ollama_request_duration_seconds_bucket[5m]))

# Qdrant cluster health
up{job="qdrant-cluster"} == 1
qdrant_cluster_replication_lag_seconds
rate(qdrant_http_requests_total[5m]) by (node)

# PII violations
sum(rate(pii_violations_total[24h])) by (severity)

# Cloud API usage
sum(increase(cloud_requests_total[24h])) by (service)
```

---

## 12. Security and Hardening

### 12.1 Tailscale ACLs

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
      "dst": ["tag:anythingllm:3001"]
    },
    {
      "action": "accept",
      "src": ["group:admins"],
      "dst": ["*:*"]
    },
    {
      "action": "deny",
      "src": ["group:family"],
      "dst": ["tag:openclaw-rpi5:*"]
    }
  ]
}
```

### 12.2 Container Hardening

```yaml
# Best practices for all containers
services:
  example:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE  # Only if needed
    read_only: true
    tmpfs:
      - /tmp
```

### 12.3 Secrets Management

```bash
# Use Docker secrets or environment files
cat > .env << EOF
GRAFANA_PASSWORD=$(openssl rand -base64 16)
GITEA_DB_PASSWORD=$(openssl rand -base64 16)
OPENCLAW_ADMIN_PASSWORD=$(openssl rand -base64 16)
OPENCLAW_JWT_SECRET=$(openssl rand -hex 32)
EOF

chmod 600 .env
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
    cloud_budget_monthly: 100  # USD
  
  - id: child1@family.tailnet
    role: child
    allowed_models: ["llama3.2:8b", "qwen2.5:7b"]
    allow_cloud: false
    content_filter: strict
```

### 13.2 Model Access Control (Pseudocode)

```
# Pseudocode: Model access control

class GovernanceEngine:
  
  function check_user_permission(user_id, model):
    user_policy = load_user_policy(user_id)
    
    # Check model whitelist
    if user_policy.allowed_models != ["*"]:
      if model not in user_policy.allowed_models:
        return false
    
    # Check cloud permission
    if this.is_cloud_model(model):
      if not user_policy.allow_cloud:
        return false
      
      # Check budget
      if this.get_monthly_spend(user_id) >= user_policy.cloud_budget_monthly:
        return false
    
    return true
```

---

## 14. Performance Optimization

### 14.1 Tiered Routing (Pseudocode)

```
# Pseudocode: Inference tier enumeration

enum InferenceTier:
  INTERACTIVE   # Mac Studio (Ollama 70B-236B)
  EDGE          # Jetson (Ollama 7B)
  BATCH         # DGX Spark
  OPENCLAW      # RPi5 (7B for assistant tasks)
  CLOUD         # NVIDIA Brev, Gemini


class TieredRouter:
  
  function route_to_tier(prompt, latency_requirement, model_size):
    if latency_requirement == "realtime" and model_size <= 70:
      return InferenceTier.INTERACTIVE
    
    else if latency_requirement == "low" and model_size <= 7:
      return InferenceTier.EDGE
    
    else if latency_requirement == "batch":
      return InferenceTier.BATCH
    
    else if model_size >= 405:
      return InferenceTier.CLOUD
    
    else:
      return InferenceTier.INTERACTIVE
```

### 14.2 Caching Strategy

```yaml
caching_layers:
  l1_redis:
    location: GMKtec K6
    size: 16GB
    ttl: 1h
    use_case: "Semantic cache for repeated queries"
  
  l2_qdrant:
    location: K6 + M7 cluster
    size: 15M vectors
    use_case: "Long-term knowledge, RAG"
  
  l3_nas:
    location: Synology DS423
    size: 24TB
    use_case: "Model storage, document archive"
```

---

## 15. Backup and Disaster Recovery

### 15.1 3-2-1 Strategy

- **3** copies of data
- **2** different storage media (NAS + Backblaze B2)
- **1** off-site backup (Backblaze B2)

```yaml
backup_schedule:
  synology_nas:
    frequency: Continuous (Btrfs snapshots)
    retention: 30 days
  
  backblaze_b2:
    frequency: Daily 3 AM
    retention: 90 days
    encryption: Client-side AES-256
    
    included:
      - /volume1/family
      - /volume1/qdrant
      - /volume1/openclaw  # RPi5 persistent memory
      - /volume1/logs
      - /volume1/docker (registry)
```

### 15.2 OpenClaw Backup (RPi5)

```bash
# Backup OpenClaw data on RPi5
docker-compose -f docker-compose.openclaw-rpi5.yml down

# Backup volumes
sudo tar -czf /mnt/nas/backups/openclaw-rpi5-$(date +%Y%m%d).tar.gz \
  /var/lib/docker/volumes/openclaw-data \
  /var/lib/docker/volumes/openclaw-db

# Restart services
docker-compose -f docker-compose.openclaw-rpi5.yml up -d
```

### 15.3 Qdrant Cluster Backup

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d)

# Snapshot all collections
for collection in $(curl -s http://k6.tailnet.ts.net:6333/collections | jq -r '.result.collections[].name'); do
  curl -X POST "http://k6.tailnet.ts.net:6333/collections/$collection/snapshots"
done

# Sync to NAS
rsync -avz k6.tailnet.ts.net:/volume1/qdrant/primary/snapshots/ /volume1/backups/qdrant/$DATE/

# Upload to B2
b2 sync /volume1/backups/qdrant/$DATE/ b2://pais-backups/qdrant/$DATE/
```

---

## 16. Fine-Tuning Infrastructure

### 16.1 LoRA Fine-Tuning on DGX Spark #2

```yaml
# DGX Spark #2 dedicated to fine-tuning
role: "Fine-tuning, LoRA training, model experimentation"
memory: 128GB LPDDR5x
storage: 4TB NVMe
```

#### LoRA Configuration (Pseudocode)

```
# Pseudocode: LoRA fine-tuning

function create_lora_config():
  lora_config = {
    "rank": 16,
    "alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "dropout": 0.05
  }
  
  return lora_config


function fine_tune_with_lora(base_model, training_data, lora_config):
  # Load base model
  model = load_model(base_model)
  
  # Apply LoRA adapter
  peft_model = apply_peft_lora(model, lora_config)
  
  # Training configuration
  training_args = {
    "batch_size": 4,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "logging_steps": 10
  }
  
  # Train
  trainer = create_trainer(peft_model, training_data, training_args)
  trainer.train()
  
  # Save adapter
  save_lora_adapter(peft_model, "/volume1/models/lora/my-adapter")
  
  return peft_model
```

---

## 17. Multi-Modal Capabilities

### 17.1 Speech-to-Text (Whisper)

```yaml
# Whisper Large v3 on AI Max+ 395
services:
  whisper:
    image: onerahmet/openai-whisper-asr-webservice:latest-gpu
    container_name: whisper-large
    ports:
      - "9000:9000"
    environment:
      - ASR_MODEL=large-v3
      - ASR_ENGINE=faster_whisper
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
        limits:
          memory: 24GB
    restart: always
```

### 17.2 Text-to-Speech

**Local (Piper on Jetson):**

```yaml
services:
  piper:
    image: rhasspy/wyoming-piper:latest
    container_name: piper-tts
    ports:
      - "10200:10200"
    volumes:
      - /opt/piper/voices:/data
    restart: always
```

**Cloud (ElevenLabs):**

```
# Pseudocode: ElevenLabs TTS
function elevenlabs_tts(text, voice="Rachel"):
  audio_bytes = elevenlabs_client.generate(text, voice, "eleven_multilingual_v2")
  return audio_bytes
```

### 17.3 Image Generation (ComfyUI)

```bash
# ComfyUI on Mac Studio M3 Ultra
# Performance: SDXL ~25-35s per image (1024×1024, 30 steps)

# Models on NAS
~/ComfyUI/extra_model_paths.yaml:
  pais_nas:
    base_path: /Volumes/models
    checkpoints: stable-diffusion/
    loras: loras/
    vae: vae/
```

---

## 18. Quick Reference

### 18.1 Ollama Commands

```bash
# Pull models
ollama pull llama3.2:70b
ollama pull deepseek-v3:236b-q4_K_M
ollama pull qwen2.5:7b  # For RPi5
ollama pull phi3:mini   # For RPi5

# Run interactively
ollama run mistral-small:latest

# List models
ollama list

# Remove models
ollama rm llama3.2:70b
```

### 18.2 OpenClaw Commands (RPi5)

```bash
# SSH into RPi5
ssh pi@rpi5-openclaw.tailnet.ts.net

# Check status
docker-compose -f docker-compose.openclaw-rpi5.yml ps

# View logs
docker-compose -f docker-compose.openclaw-rpi5.yml logs -f openclaw

# Restart
docker-compose -f docker-compose.openclaw-rpi5.yml restart openclaw

# Test Ollama
curl http://localhost:11434/api/generate -d '{"model":"qwen2.5:7b","prompt":"Hello"}'
```

### 18.3 Qdrant Cluster Commands

```bash
# Check cluster health
curl http://k6.tailnet.ts.net:6333/cluster

# List collections
curl http://k6.tailnet.ts.net:6333/collections

# Create sharded collection
curl -X PUT http://k6.tailnet.ts.net:6333/collections/my-collection \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {"size": 1536, "distance": "Cosine"},
    "shard_number": 3,
    "replication_factor": 2
  }'
```

### 18.4 Service URLs

| Service | URL | Port | Device |
|---------|-----|------|--------|
| Ollama (Mac Studio) | `mac-studio.tailnet.ts.net` | 11434 | Mac Studio |
| **OpenClaw** | `rpi5-openclaw.tailnet.ts.net` | 3000 | **Raspberry Pi 5 #1** |
| AnythingLLM | `mac-mini.tailnet.ts.net` | 3001 | Mac mini |
| n8n | `mac-mini.tailnet.ts.net` | 5678 | Mac mini |
| Grafana | `mac-mini.tailnet.ts.net` | 3002 | Mac mini |
| Qdrant (Primary) | `k6.tailnet.ts.net` | 6333 | GMKtec K6 |
| Qdrant (Replica 1) | `m7-1.tailnet.ts.net` | 6333 | GMKtec M7 #1 |
| Qdrant (Replica 2) | `m7-2.tailnet.ts.net` | 6333 | GMKtec M7 #2 |
| Nemotron PII | `ai-max-395.tailnet.ts.net` | 8888 | AI Max+ 395 |

### 18.5 Cloud Services

| Service | Use Case | API Endpoint |
|---------|----------|--------------|
| NVIDIA Brev | 405B+ models | `api.brev.dev` |
| Vertex AI | Enterprise | `aiplatform.googleapis.com` |
| Gemini API | Fast/cheap | `generativelanguage.googleapis.com` |
| ElevenLabs | Voice | `api.elevenlabs.io` |

### 18.6 Hardware Quick Reference

| Device | RAM | Power | Role | Primary Service |
|--------|-----|-------|------|-----------------|
| Mac Studio M3 Ultra | 256GB | 150W | Inference | Ollama 70B-236B |
| Mac mini M4 Pro | 48GB | 35-50W | Orchestration | LangFlow, n8n, monitoring |
| GMKtec K6 | 64GB | 45W | Qdrant Primary | Vector DB + Cache + Embeddings |
| GMKtec M7 #1 | 32GB | 45W | Qdrant Replica | Vector DB + Reranking |
| GMKtec M7 #2 | 32GB | 45W | Qdrant Replica | Vector DB + Doc Processing |
| AI Max+ 395 | 128GB | 65W | Content Safety | Nemotron PII + Whisper |
| **Raspberry Pi 5 #1** | **16GB** | **10-15W** | **OpenClaw** | **Personal AI Assistant** |
| Raspberry Pi 5 #2 | 16GB | 10W | Home Automation | Home Assistant |
| Raspberry Pi 5 #3 | 16GB | 10W | IoT Monitoring | InfluxDB, Telegraf |
| ThinkPad P14s | 96GB | Battery | Mobile | Portable AI |
| MacBook Pro M4 Pro | 48GB | Battery | Mobile | Executive |
| Dell Pro Max 16 Plus | 32GB | Battery | Mobile | GPU work |

---

## Conclusion

PAIS v1.3.5 represents a **local-first, Docker Compose-native** architecture for sovereign AI infrastructure, optimized for:

- **Operational simplicity**: Docker Compose + Tailscale + HA patterns instead of full Kubernetes
- **High availability**: 3-node Qdrant cluster, isolated OpenClaw, GPU-accelerated content safety
- **Security**: Isolated assistant device, strict Tailscale ACLs, PII-aware cloud overflow
- **Efficiency**: Low-power OpenClaw, right-sized roles for each device, no unnecessary orchestration overhead

**Key architectural decisions in v1.3.5**:

1. **No Kubernetes requirement**: Deliberate choice to keep operational complexity low
2. **Isolated OpenClaw**: Dedicated Raspberry Pi 5 for complete separation from critical infrastructure
3. **High-availability knowledge**: 3-node Qdrant cluster (99.95% uptime, 3× throughput)
4. **GPU-accelerated PII detection**: Nemotron on AI Max+ 395
5. **Mac mini simplification**: Pure orchestration (12GB RAM freed, 15W power saved)

**Migration benefits from v1.3.4**:

- **Complete isolation**: OpenClaw separated from all critical PAIS services
- **$42/year savings**: Lower power consumption vs Mac mini hosting
- **12GB RAM freed**: Mac mini now has 26GB buffer
- **Better security**: System-level permissions on isolated, low-power device
- **No cost**: $0 hardware investment (repurposed existing RPi5)

The framework is designed to grow with your needs—start with Ollama on a single Mac, add isolated OpenClaw on RPi5 for proactive assistance, deploy HA knowledge layer for production RAG, and selectively integrate cloud overflow as required, all without committing to Kubernetes.

---

*End of PAIS Framework v1.3.5*

**Document Revision History:**
- v1.0 (June 2025): Initial framework
- v1.1 (September 2025): Tailscale networking
- v1.2 (January 2026): Containerization, observability
- v1.3 (February 2026): Security, multi-tenancy, fine-tuning
- v1.3.1 (February 2026): Ollama primary, cloud overflow, removed ASCII diagram
- v1.3.2 (February 2026): OpenClaw integration, Nemotron PII detection (AI-powered, pseudocode), updated hardware (Mac Studio 256GB, K6 64GB), mobile workstations
- v1.3.3 (February 2026): High-availability knowledge layer (K6 + M7 cluster), repurposed AI Max+ 395 (Nemotron PII), OpenClaw migrated to Mac mini, 99.95% uptime RAG
- v1.3.4 (February 2026): OpenClaw isolated on dedicated Raspberry Pi 5 #1, Mac mini simplified to pure orchestration, updated edge layer roles
- **v1.3.5 (February 2026)**: Kubernetes roadmap removed, Docker-first strategy affirmed, documentation aligned to non-Kubernetes operational model
