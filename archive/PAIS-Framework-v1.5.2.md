# PRIVATE AI SOVEREIGNTY (PAIS)

**A Framework for Consumer Control of Local AI Infrastructure**

**Version 1.5.2**  
**February 2026**

*Includes: Unified Knowledge Layer (disk-based), Family Learning Features, Visual AI Hub (RTX 4090), Isolated OpenClaw on Raspberry Pi 5, Storage Architecture, Tailscale Networking, Containerization Strategy, Observability, Security Hardening, Local-First with Cloud Overflow, Multi-Tenancy, Flexible Orchestration, and Windows Support Nodes*

---

> *"If I build the tools, I have only myself to blame for the exposure and consequences of my and my family's use."*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)  
2. [What's New in Version 1.5.2](#2-whats-new-in-version-152)  
3. [Functional Specifications](#3-functional-specifications)  
4. [Functional Requirements Validation](#4-functional-requirements-validation)  
5. [Hardware Architecture](#5-hardware-architecture)  
6. [Orchestration Layer](#6-orchestration-layer)  
7. [Inference Architecture](#7-inference-architecture)  
8. [Visual AI Hub (RTX 4090)](#8-visual-ai-hub-rtx-4090)  
9. [OpenClaw Personal AI Assistant](#9-openclaw-personal-ai-assistant)  
10. [Unified Knowledge Layer](#10-unified-knowledge-layer)  
11. [Family Learning & Education Features](#11-family-learning--education-features)  
12. [Cloud Overflow Strategy](#12-cloud-overflow-strategy)  
13. [Containerization Strategy](#13-containerization-strategy)  
14. [Source Code Management](#14-source-code-management)  
15. [Observability Stack](#15-observability-stack)  
16. [Security and Hardening](#16-security-and-hardening)  
17. [Multi-Tenancy and Governance](#17-multi-tenancy-and-governance)  
18. [Performance Optimization](#18-performance-optimization)  
19. [Backup and Disaster Recovery](#19-backup-and-disaster-recovery)  
20. [Fine-Tuning Infrastructure](#20-fine-tuning-infrastructure)  
21. [Multi-Modal Capabilities](#21-multi-modal-capabilities)  
22. [Quick Reference](#22-quick-reference)  

---

## 1. Executive Summary

### 1.1 The Problem

Artificial intelligence has become essential infrastructure for knowledge work, creativity, and daily life. Yet the dominant delivery model—cloud-based APIs—creates fundamental tensions:

- **Loss of Control**: Terms of service change unilaterally; pricing increases without notice; capabilities are deprecated or restricted.  
- **Privacy Erosion**: Every prompt, document, and query transits third-party infrastructure with opaque data handling.  
- **Rising Costs**: Token-based pricing creates unpredictable expenses that scale with usage.  
- **Dependency Risk**: Critical workflows become hostage to provider availability, policy changes, and geopolitical factors.  
- **Educational Constraints**: Commercial AI services lack the pedagogical controls, privacy protections, and customization needed for family learning environments.  
- **Visual AI Dependency**: Image generation, video creation, and multimedia tools locked behind proprietary SaaS (Midjourney, OpenAI DALL-E, stock photo subscriptions).

### 1.2 The PAIS Solution

Private AI Sovereignty (PAIS) is a framework for individuals and families to establish **local-first AI infrastructure with intelligent cloud overflow** that serves as:

- **A private learning cloud** for adaptive, mastery-based education
- **A creative studio** for multi-modal content generation (text, image, video, audio)
- **A build farm** for CI/CD and development workflows
- **A memory system** for long-term knowledge management
- **A digital homestead** for family-scale sovereignty

By leveraging modern local hardware, open-source models, containerized services, secure mesh networking, and flexible orchestration tools, PAIS enables:

- **Local-first inference** — Ollama and local models handle 95%+ of workloads on-device.  
- **Multi-modal generation** — ComfyUI, Stable Diffusion, video synthesis, voice synthesis all local.
- **Intelligent cloud overflow** — Seamlessly route to NVIDIA Brev, Google Vertex AI, Gemini when needed.  
- **Full data sovereignty** — Explicit control over when data leaves your network.  
- **Adaptive learning** — Mastery-based progression, Socratic tutoring, multi-level explanations.
- **Family publishing** — Static sites, blogs, portfolios with AI-generated imagery and video.
- **Safe experimentation** — Local sandboxes for coding, data science, AI prompting without cloud risks.
- **Background AI utility** — Automated content tagging, summarization, safety checking via scheduled tasks.
- **Isolated proactive assistance** — OpenClaw on dedicated Raspberry Pi 5.  
- **Unified knowledge layer** — Single-node Qdrant on AI Max+ 395 (128GB) using disk-based HNSW.  
- **Separated AI specialization** — Mac Studio for text LLMs, RTX 4090 for visual AI.
- **Customizable governance** — Per-user policies, content safety, cloud usage controls.  
- **Secure remote access** — Tailscale mesh VPN without internet exposure.  
- **Portable architecture** — Docker containers enable workload mobility.  
- **Observable systems** — Prometheus and Grafana metrics.  
- **Windows-native support** — M7 nodes for static sites, CI/CD, dev tooling.  

---

## 2. What's New in Version 1.5.2

Version 1.5.2 integrates the **NVIDIA RTX 4090 workstation as the Visual AI Hub**, creating a **specialized multi-modal inference architecture** with clear separation of concerns:

### 2.1 New Hardware: RTX 4090 Visual AI Hub

- **Role**: Dedicated image, video, and audio generation
- **GPU**: NVIDIA RTX 4090 (24GB GDDR6X, 16,384 CUDA cores, 512 Tensor cores)
- **Services**: ComfyUI, Stable Video Diffusion, Coqui TTS, FFmpeg NVENC
- **Power**: ~600W (GPU + system)
- **Integration**: Intelligent router on Mac mini directs requests to 4090 vs Mac Studio

### 2.2 Architectural Improvements

**Separation of Concerns**:
- **Mac Studio** (256GB unified RAM): Text LLM inference (Ollama, Socratic tutor, RAG)
- **RTX 4090** (24GB VRAM): Visual AI (image, video, audio generation)
- **AI Max+ 395** (128GB): Knowledge, safety, progress tracking
- **DGX Spark**: Batch inference and fine-tuning

**Benefits**:
- ✅ **Optimal hardware utilization**: Each device does what it's built for
- ✅ **Better throughput**: Mac Studio + 4090 can run simultaneously
- ✅ **Power efficiency**: 4090's NVENC codec acceleration vs generic CPU transcoding
- ✅ **Family content generation**: Portfolio images, educational videos, voice narration
- ✅ **Eliminates SaaS**: No Midjourney, DALL-E, or stock photo subscriptions needed

### 2.3 Family Use Cases Enabled

**Visual Learning Materials**:
- Generate custom diagrams for lessons (8 seconds per image)
- Create educational animations and videos
- Produce audiobook narration for multi-sensory learning

**Portfolio Generation**:
- Custom cover art for blog posts
- Illustrated project documentation
- Portfolio-quality images without stock photo fees

**Creative Projects**:
- Kids' game assets (texture generation, sprite creation)
- 3D modeling and ray-traced renders
- Animated explainer videos

**Content Optimization**:
- Fast H.265 video transcoding via NVENC
- Batch image processing (resize, optimize, watermark)
- Audio normalization and enhancement

---

## 3. Functional Specifications

*(Identical to v1.5.1 — complete 12,000+ word specification section — see v1.5.1 Section 3)*

Includes:
- **Core System Layout** (three-zone architecture)
- **Educational Features** (12 learning-focused capabilities)
- **Productivity Features** (8 adult-focused capabilities)
- **Long-term Vision** (family-scale research lab)

---

## 4. Functional Requirements Validation

*(Identical to v1.5.1 — 20/20 requirements coverage — see v1.5.1 Section 4)*

---

## 5. Hardware Architecture

### 5.1 Reference Hardware Stack

#### Orchestration Layer

```yaml
Mac mini M4 Pro:
  cpu: 14-core (10P + 4E)
  unified_memory: 48GB
  neural_engine: 16-core
  storage: 1TB SSD
  power: ~50W
  qty: 1
  role: "Orchestration, monitoring, intelligent routing"
  
  services:
    - LangFlow (visual workflows + Socratic tutor) [4GB]
    - LMStudio (local model testing) [2GB]
    - AnythingLLM (RAG/chat + learning interface) [4GB]
    - n8n (automation + background AI tasks) [4GB]
    - Prometheus & Grafana (observability + learning analytics) [4GB]
    - API Gateway (Traefik) + Intelligent Router [2GB]
      note: "Routes text requests to Mac Studio, visual requests to 4090"
    - Model Router (intelligent tiering + cloud overflow) [2GB]
  
  total_allocation: ~22GB
  remaining: ~26GB buffer
```

#### Inference Layer – Text LLMs

```yaml
Mac Studio M3 Ultra:
  unified_memory: 256GB
  cpu: 32-core (24P + 8E)
  gpu: 80-core
  neural_engine: 32-core
  storage: 4TB SSD
  power: ~150W
  qty: 1
  role: "Primary text LLM inference via Ollama"
  
  inference_engine: Ollama
  supported_models:
    - Llama 3.2 70B (Q4_K_M) [Socratic tutor, deep reasoning]
    - Qwen 2.5 72B (Q4_K_M) [Explain-It-Three-Ways, AI editing]
    - DeepSeek v3 236B (Q4_K_M) [Research, complex reasoning]
    - Mistral Small 22B [Fast chat, journaling prompts]
    - Whisper Medium [Transcription for journals]
  
  note: "Image/video generation routed to RTX 4090"
```

#### Inference Layer – Visual AI (RTX 4090) **[NEW IN v1.5.2]**

```yaml
RTX 4090 Workstation:
  gpu: NVIDIA RTX 4090 (24GB GDDR6X)
  cuda_cores: 16,384
  tensor_cores: 512 (4th generation)
  cpu: Intel Core i9 / AMD Ryzen 9 (assumed high-end)
  system_memory: 64GB DDR5 (assumed)
  storage: 2TB NVMe SSD
  power: ~600W (450W GPU + 150W system)
  qty: 1
  role: "Multi-modal content generation: Image, Video, Audio"
  
  primary_services:
    - ComfyUI (Stable Diffusion XL, SDXL Turbo, SD3):
        allocation: 8GB VRAM
        throughput: 1 image (1024×1024) per 8-12 seconds
        use_cases:
          - Generate portfolio images
          - Create educational diagrams
          - Illustrate blog posts
          - Generate learning material graphics
    
    - Stable Video Diffusion & AnimateDiff:
        allocation: 12GB VRAM
        throughput: 15-second video in 60-90 seconds
        use_cases:
          - Educational animation videos
          - Animated explainers for concepts
          - Portfolio project videos
    
    - Coqui TTS (Voice Synthesis):
        allocation: 2GB VRAM
        throughput: Real-time or faster
        use_cases:
          - Audiobook narration for lessons
          - Multilingual lesson narration
          - Accessibility (text-to-speech for PDFs)
    
    - FFmpeg with NVENC (H.265 video encoding):
        allocation: 2GB VRAM
        throughput: Real-time or faster
        use_cases:
          - Fast video transcoding for NAS storage
          - Convert 4K to optimized streaming format
          - Batch compress family video library
    
    - LoRA Fine-tuning (SD or LLM adapters):
        allocation: 16-20GB VRAM (can run concurrent with one service)
        use_cases:
          - Fine-tune SDXL on family art style
          - Create custom Stable Diffusion "family look"
          - Train personal LLM adapters (DreamBooth alternative)
    
    - Optional: Blender with OptiX rendering:
        allocation: 24GB VRAM (dedicated)
        use_cases:
          - 3D rendering for projects
          - Real-time ray tracing for visualization
  
  hardware_rationale:
    - 24GB VRAM perfect for SDXL + LoRA fine-tuning simultaneously
    - CUDA 8.9 compute capability optimized for diffusion models
    - Tensor Cores provide 4× speedup vs Mac GPU for matrix operations
    - NVENC hardware encoder (7th gen) enables fast video processing
    - Specialization: Mac Studio stays on text LLMs (its strength)
```

#### Batch Inference & Training Layer

```yaml
DGX Spark #1 (×2):
  memory: 128GB LPDDR5x each
  storage: 4TB NVMe
  role: "Batch inference, high-concurrency text LLM service"

DGX Spark #2:
  memory: 128GB LPDDR5x
  storage: 4TB NVMe
  role: "Fine-tuning for personalized learning models"
```

#### Unified Knowledge & Content Safety Layer

```yaml
GMKtec AI Max+ 395:
  cpu: Ryzen AI 9 395 (16-core, 5.1 GHz boost)
  unified_memory: 128GB LPDDR5X
  npu: 40+ TOPS AI accelerator
  storage: 2TB PCIe 4.0 NVMe (7,000 MB/s read, 1M+ IOPS)
  power: ~75W
  qty: 1
  role: "Knowledge, safety, and user progress tracking"
  
  primary_services:
    - Qdrant vector database:
        collections:
          - learning_content (curriculum, lessons)
          - user_progress (per-user mastery tracking)
          - obsidian_vault (indexed knowledge base)
          - journals (reflection entries)
        ram_allocation: 100GB (HNSW + hot vectors)
        disk_allocation: ~1.5TB (cold vectors + indexes)
    - Redis semantic cache: 16GB
    - Embeddings: nomic-embed-text (4GB, NPU-accelerated)
    - Nemotron-4 Content Safety: 64GB (time-shared)
    - Whisper Large v3: 24GB (time-shared with Nemotron)
```

#### Development & Edge Layer

```yaml
GMKtec K6:
  cpu: AMD Ryzen 7 7840HS (8-core, up to 5.1 GHz)
  memory: 64GB DDR5
  storage: 1TB PCIe 4.0 SSD
  power: ~45W
  qty: 1
  role: "Dev, staging, JupyterLab, experiments"
  
  primary_services:
    - JupyterLab (coding sandboxes, data science) [16GB]
    - LangFlow staging [4GB]
    - n8n staging [4GB]
    - AnythingLLM staging [4GB]
    - Ollama edge inference (Qwen 2.5 14B–22B) [16GB]
    - Model A/B testing [8GB]
    - CI/CD runner (Linux jobs) [4GB]
    - System overhead [4GB]
  
  total_allocation: ~60GB
  remaining: ~4GB
```

#### Windows Support Nodes

```yaml
GMKtec M7 #1 (Windows 11 Pro):
  memory: 16GB DDR5
  power: ~25–30W
  role: "Static web hosting, family publishing platform"
  
  primary_services:
    - IIS 10 (Jekyll static sites) [~2GB]
    - Jekyll build pipeline (WSL2) [~1GB]
    - Internal wiki (Wiki.js) [~2GB]
    - Vaultwarden (password manager) [~1GB]
    - SMB file shares (portfolios, media) [~1GB]
    - Print server [~0.5GB]
    - Scheduled tasks (Jekyll builds, backups) [<0.5GB]
    - Windows overhead [~5GB]

GMKtec M7 #2 (Windows 11 Pro):
  memory: 16GB DDR5
  power: ~25–30W
  role: "Dev tools, CI/CD, build automation"
  
  primary_services:
    - Jenkins/GitLab Runner (CI/CD) [~4GB]
    - VS Code Server (web IDE) [~2GB]
    - Node.js / Python toolchain [~2GB]
    - Lightweight Docker Desktop [2–4GB]
    - SSH/RDP access [<0.5GB]
    - Artifact storage [~1GB]
    - Windows overhead [~5GB]
```

#### OpenClaw Layer

```yaml
Raspberry Pi 5 #1:
  memory: 16GB LPDDR4X
  power: 10–15W
  role: "OpenClaw 24/7 proactive assistant (isolated)"
  
  services:
    - OpenClaw (email, calendar, system monitoring) [4GB]
    - Ollama (qwen2.5:7b) [6–8GB]
    - PostgreSQL (assistant memory) [2GB]
```

#### Storage Layer

```yaml
Synology DS423 NAS:
  disks: 4× 6TB IronWolf Pro
  raid: SHR + Btrfs
  total_capacity: ~24TB usable
  power: ~45W
  
  volumes:
    /volume1/models: "Ollama and embedding models"
    /volume1/qdrant: "Qdrant snapshots + disk-based data"
    /volume1/docker: "Registry, container images"
    /volume1/git: "Source mirrors, family repos"
    /volume1/backups: "System backups, snapshots"
    /volume1/family: "Documents + encrypted cloud backup"
    /volume1/openclaw: "OpenClaw persistent memory"
    /volume1/obsidian: "Personal knowledge bases"
    /volume1/portfolios: "Family digital portfolios (Jekyll sources)"
    /volume1/jupyter: "JupyterLab notebooks"
    /volume1/learning: "Progress data, journals, reflections"
    /volume1/generated: "AI-generated images, videos, audio" **[NEW]**
    /volume1/comfyui: "ComfyUI models, workflows, outputs" **[NEW]**
```

---

## 6. Orchestration Layer

*(Unchanged from v1.5.1)*

### 6.1 Intelligent Request Router (NEW in v1.5.2)

**Router Logic** (pseudo-code on Mac mini):

```python
class IntelligentRequestRouter:
    """Routes requests to optimal inference backend"""
    
    def route_request(self, request):
        request_type = request.get('type')  # 'text', 'image', 'video', 'audio'
        
        if request_type == 'text':
            # Route to Mac Studio (256GB unified memory, optimized for text)
            return {
                'backend': 'ollama-mac-studio',
                'url': 'http://mac-studio.tailnet.ts.net:11434',
                'models': ['llama3.2:70b', 'qwen2.5:72b', 'deepseek-v3:236b']
            }
        
        elif request_type == 'image':
            # Route to RTX 4090 (24GB VRAM, optimized for diffusion)
            return {
                'backend': 'comfyui-4090',
                'url': 'http://rtx-4090.tailnet.ts.net:8188',
                'models': ['SDXL', 'SDXL-Turbo', 'SD3']
            }
        
        elif request_type == 'video':
            # Route to RTX 4090 (NVENC encoder + diffusion)
            return {
                'backend': 'stable-video-4090',
                'url': 'http://rtx-4090.tailnet.ts.net:7860',
                'throughput': '15s video in 60-90 seconds'
            }
        
        elif request_type == 'audio':
            # Route to RTX 4090 (Coqui TTS)
            return {
                'backend': 'coqui-4090',
                'url': 'http://rtx-4090.tailnet.ts.net:5002',
                'features': ['multilingual', 'voice-cloning', 'realtime']
            }
        
        elif request_type == 'video_transcode':
            # Route to RTX 4090 (NVENC hardware encoder)
            return {
                'backend': 'ffmpeg-nvenc-4090',
                'url': 'http://rtx-4090.tailnet.ts.net:9000',
                'codec': 'H.265 (HEVC)',
                'throughput': 'Real-time or faster'
            }
        
        else:
            # Default: Mac Studio
            return {'backend': 'ollama-mac-studio'}
```

### 6.2 Updated Service URLs

| Service | URL | Purpose | Backend |
|---------|-----|---------|---------|
| LangFlow | `mac-mini.tailnet.ts.net:7860` | Workflow builder | Mac mini |
| AnythingLLM | `mac-mini.tailnet.ts.net:3001` | Chat interface | Mac mini (routes to backends) |
| Ollama (Text) | `mac-studio.tailnet.ts.net:11434` | Text LLM API | Mac Studio |
| **ComfyUI (Image)** | **`rtx-4090.tailnet.ts.net:8188`** | **Image generation** | **RTX 4090** |
| **Stable Video** | **`rtx-4090.tailnet.ts.net:7860`** | **Video generation** | **RTX 4090** |
| **Coqui TTS** | **`rtx-4090.tailnet.ts.net:5002`** | **Voice synthesis** | **RTX 4090** |
| **FFmpeg NVENC** | **`rtx-4090.tailnet.ts.net:9000`** | **Video transcode** | **RTX 4090** |
| n8n | `mac-mini.tailnet.ts.net:5678` | Automation | Mac mini |
| Grafana | `mac-mini.tailnet.ts.net:3002` | Monitoring | Mac mini |

---

## 7. Inference Architecture

### 7.1 Dual-Specialization Model

**Mac Studio**: Text LLM inference (Ollama)

```bash
# Install Ollama on Mac Studio
brew install ollama

# Available models (256GB sufficient for all)
ollama pull llama3.2:70b        # Socratic tutor
ollama pull qwen2.5:72b         # General chat, explanations
ollama pull deepseek-v3:236b    # Deep reasoning
ollama pull mistral-small:22b   # Fast responses
```

**RTX 4090**: Visual AI inference (ComfyUI, Stable Video, Coqui)

```bash
# Install ComfyUI on RTX 4090 workstation
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install -r requirements.txt
python main.py --listen 0.0.0.0  # Listen on Tailscale

# Install Stable Video Diffusion
# Install Coqui TTS
# Install FFmpeg with NVENC support
```

### 7.2 Model Performance Expectations

| Device | Task | Model | Throughput | Memory |
|--------|------|-------|-----------|--------|
| Mac Studio | Text | Llama 3.2 70B | 6–7 tok/s | ~45GB |
| Mac Studio | Text | Qwen 2.5 72B | 5–6 tok/s | ~46GB |
| Mac Studio | Text | DeepSeek v3 236B | 2–3 tok/s | ~155GB |
| **RTX 4090** | **Image** | **SDXL 1.0** | **8–12 sec** | **~8GB VRAM** |
| **RTX 4090** | **Image** | **SDXL-Turbo** | **1–2 sec** | **~8GB VRAM** |
| **RTX 4090** | **Video** | **Stable Video** | **60–90 sec** | **~12GB VRAM** |
| **RTX 4090** | **Audio** | **Coqui TTS** | **Real-time** | **~2GB VRAM** |
| **RTX 4090** | **Transcode** | **H.265 via NVENC** | **Real-time+** | **~2GB VRAM** |

---

## 8. Visual AI Hub (RTX 4090)

### 8.1 ComfyUI Setup & Deployment

**Docker Deployment**:

```yaml
# docker-compose.comfyui.yml on RTX 4090 workstation
version: '3.8'

services:
  comfyui:
    image: comfyanonymous/comfyui:latest-nvidia
    container_name: comfyui
    ports:
      - "8188:8188"
    volumes:
      - /volume1/comfyui/models:/root/ComfyUI/models
      - /volume1/comfyui/outputs:/root/ComfyUI/output
      - /volume1/comfyui/input:/root/ComfyUI/input
    environment:
      - CUDA_VISIBLE_DEVICES=0  # RTX 4090
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']  # GPU 0 (4090)
              capabilities: [gpu, compute, utility]
        limits:
          memory: 32GB  # System RAM limit
    restart: always
    networks:
      - tailscale
```

**Models to Download**:

```bash
# SDXL 1.0 (base model)
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0

# SDXL Refiner (optional, for ultra-quality)
huggingface-cli download stabilityai/stable-diffusion-xl-refiner-1.0

# SDXL Turbo (1-2 step generation, real-time)
huggingface-cli download stabilityai/sdxl-turbo

# ControlNet modules (for conditional generation)
# Upscalers for high-res output
# VAE encoders for better quality
```

### 8.2 Stable Video Diffusion Deployment

```yaml
services:
  stable-video:
    image: stabilityai/stable-video-diffusion:latest
    container_name: stable-video
    ports:
      - "7860:7860"
    volumes:
      - /volume1/comfyui/models:/models
      - /volume1/generated/videos:/outputs
    environment:
      - MODEL_PATH=/models/svd-xt.safetensors
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
        limits:
          memory: 32GB
    restart: always
```

**Usage Example** (Python):

```python
import requests
import json

# Generate educational animation
payload = {
    "image_path": "/volume1/learning/photosynthesis_diagram.png",
    "motion_bucket_id": 127,
    "fps": 25,
    "num_inference_steps": 25,
    "decode_chunk_size": 8,
    "num_frames": 25  # 1 second at 25fps
}

response = requests.post(
    "http://rtx-4090.tailnet.ts.net:7860/api/generate",
    json=payload
)

video_path = response.json()['video_path']
print(f"Video generated: {video_path}")
```

### 8.3 Coqui TTS (Voice Synthesis) Deployment

```yaml
services:
  coqui-tts:
    image: coqui/tts:latest-gpu
    container_name: coqui-tts
    ports:
      - "5002:5002"
    volumes:
      - /volume1/models/tts:/root/.local/share/tts_models
      - /volume1/generated/audio:/outputs
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - TTS_HOME=/root/.local/share/tts_models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
        limits:
          memory: 8GB
    restart: always
```

**Usage Example** (Python):

```python
import requests

# Generate audiobook narration
payload = {
    "text": "Today we will learn about photosynthesis. Plants convert sunlight into energy...",
    "language": "en",
    "speaker": "female",
    "speed": 1.0,
    "emotion": "calm"
}

response = requests.post(
    "http://rtx-4090.tailnet.ts.net:5002/tts",
    json=payload
)

audio_path = response.json()['audio_path']
print(f"Audio generated: {audio_path}")
```

### 8.4 FFmpeg NVENC Video Transcoding

**Standalone Service** (via Docker):

```bash
docker run -d \
  --name ffmpeg-nvenc \
  --gpus all \
  -v /volume1/family/videos:/input \
  -v /volume1/generated:/output \
  -p 9000:8000 \
  jrottenberg/ffmpeg:latest-nvidia \
  sh -c "while true; do sleep 3600; done"

# Usage: Transcode 4K to optimized H.265
ffmpeg -i /input/family_4k.mp4 \
  -c:v hevc_nvenc \
  -preset medium \
  -crf 23 \
  -c:a aac \
  -b:a 128k \
  /output/family_hd.mp4

# Throughput: Real-time or faster
# 1 hour video transcoded in ~45 minutes
```

### 8.5 LoRA Fine-Tuning on 4090

**Fine-tune SDXL on family art style**:

```python
# Using diffusers library with xformers optimization
from diffusers import StableDiffusionXLPipeline
from peft import get_peft_model, LoraConfig
import torch

# Load SDXL base model
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
)

# Apply LoRA for family art style
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v"],
    lora_dropout=0.05,
    bias="none"
)

# Fine-tune on family photo dataset
# Training time: ~2 hours on RTX 4090

# Save adapter
pipeline.save_lora_weights("/volume1/models/lora/family-style-v1")
```

### 8.6 Usage Examples: Family Learning

#### **Example 1: Generate Lesson Diagram**

```python
# Teacher wants a diagram for explaining water cycle
prompt = """
Educational diagram showing the water cycle:
- Evaporation: Water rising from ocean
- Condensation: Clouds forming
- Precipitation: Rain falling
- Collection: Water returning to ocean

Style: Clean, flat design, suitable for ages 10-14
Colors: Blue, white, green
"""

# Via ComfyUI API
result = generate_image(
    prompt=prompt,
    model="SDXL",
    width=1024,
    height=768,
    steps=30,
    guidance_scale=7.5
)

# Output saved to /volume1/generated/lesson-diagrams/water-cycle.png
# Ready to embed in Jekyll blog post
```

#### **Example 2: Generate Portfolio Cover Art**

```python
# Child 2 writing essay on climate change, needs cover art
prompt = """
A visually striking illustration for an essay on renewable energy:
- Solar panels catching golden sunlight
- Wind turbines on rolling green hills
- Electric vehicles on clean roads
- People enjoying clean nature

Style: Hopeful, optimistic, digital art, vibrant colors
Aspect ratio: 16:9 (blog header)
"""

# Generate with SDXL-Turbo for speed (1-2 seconds)
result = generate_image_turbo(
    prompt=prompt,
    width=1600,
    height=900
)

# Output: /volume1/portfolios/child2/climate-essay-cover.png
# Auto-included in Jekyll post via template
```

#### **Example 3: Create Educational Animation**

```python
# Parent creating lesson video on photosynthesis
# Step 1: Generate base diagram with ComfyUI
base_diagram = generate_image(
    prompt="Detailed photosynthesis process diagram",
    model="SDXL"
)

# Step 2: Animate with Stable Video Diffusion
animated_video = generate_video(
    image=base_diagram,
    motion_intensity=0.7,  # Gentle animation
    duration=15  # seconds
)

# Step 3: Add voice narration
narration = generate_speech(
    text="Photosynthesis is how plants convert sunlight into energy...",
    voice="calm_female",
    speed=0.9  # Slightly slower for learning
)

# Step 4: Combine video + audio
final_video = merge_video_audio(
    video=animated_video,
    audio=narration
)

# Output: /volume1/generated/lessons/photosynthesis-lesson.mp4
# Embedded in AnythingLLM for interactive learning
```

#### **Example 4: Audiobook Narration**

```python
# Child 1 writes a short story, wants audiobook version
story_text = """
The Mystery of the Hidden Garden

Sarah discovered an old wooden door behind the library...
[2000 words of story]
"""

# Generate multilingual narration
english_audio = generate_speech(
    text=story_text,
    language="en",
    speaker="young_female"  # Match character
)

spanish_audio = generate_speech(
    text=translate(story_text, "es"),
    language="es",
    speaker="young_female"
)

# Output audiobooks
english_path = "/volume1/portfolios/child1/story-en.mp3"
spanish_path = "/volume1/portfolios/child1/story-es.mp3"

# Link in Jekyll portfolio
# Family members can listen to child's story in multiple languages
```

#### **Example 5: Night Batch Processing**

```python
# n8n workflow: Every night, generate learning variations
n8n_workflow = {
    "trigger": "cron at 2 AM",
    "steps": [
        {
            "name": "Get tomorrow's lesson topics",
            "query_qdrant": {
                "collection": "learning_content",
                "filter": {"date": "tomorrow"}
            }
        },
        {
            "name": "Generate visual variations",
            "for_each_topic": [
                {
                    "generate_image": {
                        "prompt": "Create 3 different visual explanations of {{ topic }}",
                        "models": ["SDXL", "SDXL-Turbo", "SD3"],
                        "output": "/volume1/generated/lesson-variations/{{ topic }}_*.png"
                    }
                }
            ]
        },
        {
            "name": "Generate narration options",
            "tts": {
                "text": "Tomorrow's lesson: {{ topic }}",
                "voices": ["calm", "energetic", "storyteller"],
                "output": "/volume1/generated/narration/{{ topic }}_*.mp3"
            }
        },
        {
            "name": "Email parent summary",
            "email": {
                "to": "parent@family.tailnet",
                "subject": "Tomorrow's learning materials ready",
                "body": "Generated 9 visual variations and 3 narrations for {{ topics }}"
            }
        }
    ]
}
```

---

## 9. OpenClaw Personal AI Assistant

*(Unchanged from v1.5.1 — Raspberry Pi 5 deployment)*

---

## 10. Unified Knowledge Layer

*(Unchanged from v1.5.1 — Qdrant hybrid RAM+disk config)*

---

## 11. Family Learning & Education Features

*(Unchanged from v1.5.1 — All learning features from Section 10 of v1.5.1)*

---

## 12. Cloud Overflow Strategy

*(Unchanged from v1.5.1 — Nemotron PII gate, Brev/Gemini routing)*

**New consideration for 4090**: 
- RTX 4090 is sufficient for 95%+ of image/video generation needs
- Only overflow to cloud if generating 1000s of images in parallel
- Coqui TTS sufficient for family voice needs (no cloud TTS required)

---

## 13. Containerization Strategy

*(Unchanged from v1.5.1 — Docker registry on Synology, with new ComfyUI/Video/TTS images)*

---

## 14. Source Code Management

*(Unchanged from v1.5.1 — GitHub primary, Synology mirror)*

---

## 15. Observability Stack

### 15.1 Updated Prometheus Configuration

```yaml
scrape_configs:
  # Existing jobs...
  
  # RTX 4090 metrics (NEW)
  - job_name: 'comfyui-4090'
    static_configs:
      - targets: ['rtx-4090.tailnet.ts.net:8188']
  
  - job_name: 'stable-video-4090'
    static_configs:
      - targets: ['rtx-4090.tailnet.ts.net:7860']
  
  - job_name: 'coqui-tts-4090'
    static_configs:
      - targets: ['rtx-4090.tailnet.ts.net:5002']
  
  - job_name: 'gpu-metrics-4090'
    static_configs:
      - targets: ['rtx-4090.tailnet.ts.net:9445']  # NVIDIA DCGM exporter
```

### 15.2 Grafana Dashboards for Visual AI

```json
{
  "dashboard": {
    "title": "Visual AI Hub (RTX 4090) Metrics",
    "panels": [
      {
        "title": "GPU Utilization",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "nvidia_smi_utilization_gpu_percent"
          }
        ]
      },
      {
        "title": "VRAM Usage",
        "type": "gauge",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "nvidia_smi_memory_used_mb / 24576 * 100"
          }
        ]
      },
      {
        "title": "Image Generation Throughput",
        "type": "stat",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "rate(comfyui_images_generated_total[5m])"
          }
        ]
      },
      {
        "title": "Average Generation Time",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, comfyui_generation_duration_seconds_bucket)"
          }
        ]
      }
    ]
  }
}
```

---

## 16. Security and Hardening

*(Unchanged from v1.5.1 — Tailscale ACLs remain same)*

**New consideration for 4090**: 
- ComfyUI API should only accept requests from authenticated Tailscale users
- Implement rate limiting on image generation (prevent abuse)
- Store generated images on encrypted NAS volume

---

## 17. Multi-Tenancy and Governance

*(Unchanged from v1.5.1 — Per-user learning profiles)*

**New consideration for 4090**:
```yaml
image_generation_budget:
  parent@family.tailnet:
    monthly_quota: unlimited
    cost_per_image: $0 (local)
  
  child1@family.tailnet:
    monthly_quota: 100 images/month (reasonable for homework)
    cost_per_image: $0 (local)
  
  child2@family.tailnet:
    monthly_quota: 200 images/month (more advanced)
    cost_per_image: $0 (local)
```

---

## 18. Performance Optimization

*(Unchanged from v1.5.1 — Tiered routing)*

**New optimization for 4090**:
```python
# Dynamic batching for image generation
batch_requests = []
max_batch_size = 4  # RTX 4090 can handle 4 concurrent requests
timeout = 10  # seconds

while waiting_requests:
    batch_requests.append(next_request)
    
    if len(batch_requests) >= max_batch_size or timeout_reached:
        # Process entire batch concurrently
        process_batch_parallel(batch_requests)
        batch_requests = []
```

---

## 19. Backup and Disaster Recovery

*(Unchanged from v1.5.1 — 3-2-1 strategy)*

**New volumes to backup**:
- `/volume1/comfyui`: ComfyUI models, workflows
- `/volume1/generated`: AI-generated images, videos, audio

---

## 20. Fine-Tuning Infrastructure

### 20.1 Personalized Visual Models (NEW in v1.5.2)

```yaml
use_case: "Fine-tune SDXL on family-specific visual style"

approach:
  - Collect family photos (50-100 curated examples)
  - Fine-tune SDXL LoRA adapter on RTX 4090
  - Deploy adapter for instant "family look" generation

example_workflow:
  - Parent: "Generate a vacation photo of us at the beach"
  - System: Applies family-style LoRA → SDXL generates
  - Output: Looks like it was taken by family photographer
  
training_cost:
  time: ~2 hours on RTX 4090
  data: ~100 images
  storage: ~500MB (LoRA weights)
```

---

## 21. Multi-Modal Capabilities

### 21.1 Complete Multi-Modal Stack (v1.5.2)

**Text LLMs** (Mac Studio):
- Ollama inference (70B-236B models)
- ChatGPT-like interface via AnythingLLM

**Image Generation** (RTX 4090):
- Stable Diffusion XL, SDXL-Turbo, SD3
- ControlNet for conditional generation
- Upscalers for high-resolution output

**Video Generation** (RTX 4090):
- Stable Video Diffusion (image → animation)
- AnimateDiff (text → animation)
- Frame interpolation for smooth motion

**Voice Synthesis** (RTX 4090):
- Coqui TTS (text → speech, multiple voices)
- Voice cloning (RVC, Real-Time Voice Clone)
- Multilingual support

**Speech-to-Text** (AI Max+ 395):
- Whisper Large v3 (journal transcription)

**Transcoding** (RTX 4090):
- FFmpeg NVENC (H.265 video compression)
- Real-time or faster throughput

---

## 22. Quick Reference

### 22.1 Complete Hardware Quick Reference (v1.5.2)

| Device | RAM/VRAM | Power | Role | Learning Features |
|--------|----------|-------|------|-------------------|
| Mac Studio M3 Ultra | 256GB | 150W | **Text LLM inference** | ✅ Socratic tutor, deep reasoning |
| **RTX 4090 Workstation** | **24GB VRAM + 64GB RAM** | **~600W** | **Visual AI Hub** | **✅ Image/video/audio generation** |
| AI Max+ 395 | 128GB | 75W | Knowledge + safety | ✅ User progress, RAG |
| Mac mini M4 Pro | 48GB | 50W | Orchestration, routing | ✅ Intelligent request routing |
| GMKtec K6 | 64GB | 45W | JupyterLab + dev | ✅ Coding sandboxes |
| GMKtec M7 #1 | 16GB | 25W | Jekyll + IIS | ✅ Family portfolios |
| GMKtec M7 #2 | 16GB | 25W | CI/CD + builds | ✅ Build automation |
| RPi5 #1 | 16GB | 10-15W | OpenClaw assistant | — |
| RPi5 #2 | 16GB | 10W | Home automation | — |
| RPi5 #3 | 16GB | 10W | IoT monitoring | — |
| Synology DS423 | — | 45W | Storage, backups | ✅ Generated content vault |

**Total core power (full stack)**: ~1,045W (vs ~425W without 4090)  
**Additional monthly cost**: ~$60 for 600W @ $0.10/kWh  

### 22.2 Visual AI Endpoints (NEW)

| Service | URL | Throughput | Use Case |
|---------|-----|-----------|----------|
| **ComfyUI** | **`rtx-4090.tailnet.ts.net:8188`** | **1 image / 8-12 sec** | Portfolio/lesson images |
| **Stable Video** | **`rtx-4090.tailnet.ts.net:7860`** | **15s video / 60-90 sec** | Educational animations |
| **Coqui TTS** | **`rtx-4090.tailnet.ts.net:5002`** | **Real-time** | Voice narration, audiobooks |
| **FFmpeg NVENC** | **`rtx-4090.tailnet.ts.net:9000`** | **Real-time+** | Video transcoding, compression |

### 22.3 Sample Multi-Modal Workflows (v1.5.2)

**1. Complete Educational Content Pipeline**:
```
Topic (text) → Generate diagram (ComfyUI) → Animate (Stable Video)
         → Generate narration (Coqui TTS) → Merge (FFmpeg)
         → Output: Full lesson video (no external SaaS)
```

**2. Portfolio Image Generation**:
```
Child writes blog post → Parent: "Generate cover art for this essay"
         → ComfyUI generates 3 variations → Child picks favorite
         → Auto-optimized via NVENC → Jekyll post published
```

**3. Night Batch: Generate Learning Variations**:
```
n8n cron (2 AM) → Query tomorrow's lessons → Generate:
  - 3 visual explanations per topic (ComfyUI)
  - 3 voice narrations per topic (Coqui TTS)
  - 1 animated explainer per topic (Stable Video)
→ Email parent summary → Ready for morning
```

**4. Audiobook Generation**:
```
Child's story (written) → Coqui TTS (multiple voices)
         → Narration MP3 → Link in Jekyll portfolio
         → Accessible for English learners, visually-impaired, mobile listeners
```

---

## Conclusion

PAIS v1.5.2 achieves **complete multi-modal AI sovereignty** with optimal hardware specialization:

### What PAIS v1.5.2 Delivers

1. **100% functional coverage** (20/20 requirements)
2. **Dual-specialization inference architecture**:
   - Mac Studio (256GB) → Text LLMs
   - RTX 4090 (24GB VRAM) → Visual AI
3. **Complete multi-modal capability**:
   - Text generation (Ollama)
   - Image generation (ComfyUI, SDXL)
   - Video synthesis (Stable Video, AnimateDiff)
   - Voice synthesis (Coqui TTS)
   - Video transcoding (FFmpeg NVENC)
4. **Family content generation**: Portfolios, educational videos, audiobooks, illustrations
5. **$60/month additional cost** for elimination of Midjourney, DALL-E, stock photos, video editing SaaS

### Architecture Efficiency

| Task | Before (Cloud SaaS) | After (PAIS + 4090) | Savings |
|------|-------------------|-------------------|---------|
| Generate lesson image | Midjourney ($30/mo) | ComfyUI local ($5/mo power) | $25 |
| Create animated video | Adobe After Effects + stock footage | Stable Video + Coqui TTS | $50 |
| Transcribe journal | Rev.com ($25/mo) | Whisper local | $25 |
| Generate cover art | DALL-E ($15/mo) | ComfyUI local | $15 |
| **Total monthly savings** | **$120+/month** | **~$5/month (power)** | **$115+** |

### The Result

PAIS v1.5.2 is now a **fully-sovereign, multi-modal creative studio** that:
- ✅ Runs 95%+ of AI workloads locally
- ✅ Requires zero SaaS subscriptions for core features
- ✅ Provides intelligent cloud overflow when needed
- ✅ Enables family-scale content creation
- ✅ Preserves complete data privacy
- ✅ Costs ~$60/month in electricity (all-in)

---

*End of PAIS Framework v1.5.2*

**Document Revision History:**
- v1.0 (June 2025): Initial framework  
- v1.1 (September 2025): Tailscale networking  
- v1.2 (January 2026): Containerization, observability  
- v1.3 (February 2026): Security, multi-tenancy, fine-tuning  
- v1.4.0 (February 2026): Unified knowledge layer (256GB)  
- v1.4.1 (February 2026): Unified knowledge layer (128GB, disk-based)  
- v1.4.2 (February 2026): Reintroduced M7s (Windows support nodes)  
- v1.5.0 (February 2026): Family learning & education features  
- v1.5.1 (February 2026): Added comprehensive Functional Specifications section  
- **v1.5.2 (February 2026)**: Integrated RTX 4090 Visual AI Hub, dual-specialization inference architecture (Mac Studio text + 4090 visual), complete multi-modal capabilities, family content generation workflows, intelligent request router, $60/month cost for complete AI sovereignty
