# Sovereign AI Cloud (SAIC) v1.5.3

**A Framework for Control of Local AI Infrastructure**

February 2026

---

## Overview

**Sovereign AI Cloud (SAIC)** is a comprehensive framework for individuals and families to establish **local-first AI infrastructure with intelligent cloud overflow**. It provides a "Family Learning Cloud" and "Creative Studio" that offers the benefits of modern AI—adaptive tutoring, image generation, summarization—without the privacy erosion, rising costs, and dependency risks of commercial SaaS.

### Key Principles

- **Local-First Inference**: 95%+ of workloads (text, image, audio) run on local hardware (Mac Studio, RTX 4090).
- **Intelligent Cloud Overflow**: Seamlessly routes requests to cloud providers (Google Gemini, NVIDIA Brev) only when local capacity is exceeded.
- **Data Sovereignty**: Your data (journals, photos, code) never leaves your network unless you explicitly allow it.
- **Family Learning Cloud**: Adaptive learning paths, Socratic tutoring, and safe sandboxes for kids.
- **Visual AI Hub**: Dedicated RTX 4090 node for high-speed image and video generation.
- **Dual-Network Isolation**: Strict separation between "Personal Devices" and "AI Operations" via separate Tailscale networks.

---

## Repository Contents

| File | Description |
|------|-------------|
| `SAIC-Framework-v1.5.3.md` | The complete technical specification, architecture, and user guide. |
| `SAIC-flow-animation-v1.5.3.html` | Interactive web visualization of token flow through the system. |
| `SAIC-Architecture_v1.5.3.drawio` | Detailed visual architecture diagram (draw.io format). |

---

## Architecture Summary

SAIC uses a **Three-Zone Model** across two isolated networks to ensure security and ease of use.

### 1. Personal Devices Zone (Family Tailnet)
- **Devices**: iPads, Laptops, Phones.
- **Access**: Users connect via Tailscale to a unified **Gateway** (managed by the Orchestrator).
- **User Experience**: Single web portal for Chat, Learning, and Portfolios. No CLI required.

### 2. AI Services Zone (AI Operations Tailnet)
- **Mac Studio M3 Ultra**: Primary text inference (Ollama, 70B+ models).
- **RTX 4090 Workstation**: Visual AI Hub (ComfyUI, Stable Video, Coqui TTS).
- **AI Max+ 395**: Knowledge layer (Qdrant vector DB, Nemotron safety).
- **DGX Spark Nodes**: Batch processing and fine-tuning.
- **Mac mini M4 Pro**: The **Orchestrator** and Bridge. Handles routing, governance, and low-code workflows (LangFlow, n8n).

### 3. Data Zone (Family Tailnet + Bridge)
- **Synology DS423**: Central storage for models, backups, portfolios, and journals.
- **Resilience**: 3-2-1 backup strategy with offsite encryption.

---

## Hardware Inventory (v1.5.3)

| Device | Role | Key Specs |
|--------|------|-----------|
| **Mac Studio M3 Ultra** | Text LLM Inference | 256GB Unified RAM, 60-core GPU |
| **RTX 4090 Workstation**| Visual AI Hub | 24GB VRAM, 64GB System RAM |
| **AI Max+ 395** | Knowledge & Safety | 128GB RAM, Ryzen AI 9 |
| **Mac mini M4 Pro** | Orchestration & Routing | 48GB RAM, 10GbE |
| **GMKtec K6** | Dev, Staging, Jupyter | 64GB RAM, Ryzen 7 |
| **GMKtec M7 x2** | Windows Web/CI Nodes | 16GB RAM each |
| **DGX Spark x2** | Batch Inference | 128GB RAM each |
| **Raspberry Pi 5** | OpenClaw Assistant | 16GB RAM (Isolated) |
| **Synology DS423** | Storage Layer | 24TB Usable (SHR) |

---

## Quick Start

1.  **Read the Specifications**: Start with `SAIC-Framework-v1.5.3.md` to understand the full capabilities and "User Stories".
2.  **View the Diagram**: Open `SAIC-Architecture_v1.5.3.drawio` to see the physical and logical topology.
3.  **Trace the Flow**: Open `SAIC-flow-animation-v1.5.3.html` in a browser to visualize how a request (e.g., "Explain Photosynthesis") routes through the zones.

---

## Version History

- **v1.0 - v1.3**: Initial PAIS concepts (Private AI Sovereignty).
- **v1.4**: Introduction of Unified Knowledge Layer.
- **v1.5.0**: Focus on Family Learning & Education features.
- **v1.5.2**: Added RTX 4090 Visual AI Hub.
- **v1.5.3 (Current)**: Renamed to **Sovereign AI Cloud (SAIC)**. Fully consolidated documentation. Explicit Dual-Tailnet architecture.

---

*"If I build the tools, I have only myself to blame for the exposure and consequences of my and my family's use."*
