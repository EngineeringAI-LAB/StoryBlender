<div align="center">

# 🎬 StoryBlender
### *Inter-Shot Consistent and Editable 3D Storyboard with Spatial-temporal Dynamics*

[![Project Page](https://img.shields.io/badge/🌐%20Project%20Page-StoryBlender-0A66C2?style=for-the-badge)](https://engineeringai-lab.github.io/StoryBlender/)
[![arXiv](https://img.shields.io/badge/📄%20arXiv-2604.03315-B31B1B?style=for-the-badge)](https://arxiv.org/abs/2604.03315)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-CineBoard3D-FFB000?style=for-the-badge)](https://huggingface.co/datasets/EngineeringAI-LAB/CineBoard3D)
[![Discussions](https://img.shields.io/badge/💬%20Discussions-Ask%20Questions-2EA44F?style=for-the-badge)](https://github.com/EngineeringAI-LAB/StoryBlender/discussions)

</div>

---

## ✨ Overview

**StoryBlender** is a Blender extension and web-based creative system for automated, editable, and inter-shot consistent **3D storyboard production** with spatial-temporal dynamics.

It combines:
- A native Blender add-on panel
- A Gradio-based multi-stage web UI
- MCP-style Blender communication for scene-aware operations

---

## 👥 Authors

**Bingliang Li***, **Zhenhong Sun***, **Jiaming Bian**, **Yuehao Wu**, **Yifu Wang**, **Hongdong Li**, **Yatao Bian**, **Huadong Mo†**, **Daoyi Dong**

*: Equal contribution, †: Corresponding author

---

## 🧩 Repository Structure

```text
StoryBlender/
├── __init__.py                 # Blender add-on entrypoint
├── blender_manifest.toml       # Blender extension metadata
├── wheels/                     # Bundled Python wheels for Blender
└── src/
    ├── blender_mcp.py          # Blender socket command server
    ├── server.py               # MCP bridge server
    └── gradio_app/
        ├── storyblender_app.py # Main Gradio UI
        ├── config.py           # Configuration panel and handlers
        └── step_*.py           # End-to-end storyboard workflow steps
```

---

## 🚀 Quick Start

### 1) Package the extension as a zip

Run this command from the parent directory of `StoryBlender/`:

```bash
zip -r storyblender.zip StoryBlender
```

### 2) Install in Blender

In Blender (please use Blender 4.5 LTS, 5.0+ has compatible issues), go to:

`Edit` → `Preferences` → `Get Extensions` → *(top-right dropdown menu)* → `Install from Disk`

Then select `storyblender.zip`.

Please also refer to Blender's official add-on/extension documentation:
- https://docs.blender.org/manual/en/latest/editors/preferences/addons.html

### 3) Launch the web UI

In the StoryBlender panel, click:

`Launch Gradio`

---

## ⚙️ Configuration (Required)

After launching Gradio, fill in the following fields in the **Configuration** section.

### Optional: auto-fill settings with `.env`

For easier local testing, you can keep your API keys and preferred defaults in a private `.env` file. StoryBlender loads it automatically before building the Gradio UI, so the Configuration fields are pre-filled when you launch from Blender.

```bash
cp .env.example .env
```

Then edit `.env` and fill in the keys you use. The `.env` file is ignored by git, so it is safe for local secrets as long as you do not manually add it to a commit. If you prefer to store secrets somewhere else, set `STORYBLENDER_ENV_FILE` to an absolute path before launching Blender.

Required and optional values:
- Required: `ANYLLM_API_KEY`, `ANYLLM_API_BASE`, and `ANYLLM_PROVIDER` for reasoning and vision steps.
- Required, choose one: Gemini image generation (`GEMINI_API_KEY`, `GEMINI_API_BASE`, `GEMINI_IMAGE_MODEL`) or OpenAI-compatible image generation (`OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_IMAGE_MODEL`). Select the active one with `STORYBLENDER_IMAGE_GEN_PLATFORM`.
- Optional: `SKETCHFAB_API_KEY`, required only if you want to retrieve 3D models from Sketchfab.
- Required, choose one: Meshy (`MESHY_API_KEY`, `MESHY_MODEL`), Hunyuan3D (`TENCENT_SECRET_ID`, `TENCENT_SECRET_KEY`), or local TRELLIS2 (`TRELLIS2_API_BASE` and related `TRELLIS2_*` settings) for 3D model generation. Select the active one with `STORYBLENDER_AI_PLATFORM`.
- Optional: `UTHANA_API_KEY` and `UTHANA_FPS`. If left empty, the Uthana animation pipeline and the animation auto pipeline will not be available.
- Optional: `STORYBLENDER_PROJECT_DIR`, an absolute project directory to pre-fill the UI.

### First row
- **Gemini Image Model**: name of Nano Banana (e.g. `gemini-3-pro-image-preview`, `gemini-3.1-flash-image-preview`). Note that currently Gemini is becoming very strict about copyright censorship, we are currently working on to incorporate more models to bypass this issue.
- **Gemini API Key**: your API key
- **Gemini API Base**: leave empty if you are using the official API

### Second row *(handled by AnyLLM)*
AnyLLM docs: https://mozilla-ai.github.io/any-llm/

- **Reasoning Model**: model for complex reasoning (e.g. `gemini-3.1-pro-preview`, `gpt-5.4`)
- **Vision Model**: lighter model for fast multi-model inference (e.g. `gemini-3-flash-preview`, `gpt-5.4-mini`)
- **AnyLLM API Key**: API key (can be the same as Gemini API key)
- **AnyLLM API Base**: leave empty if using official API
- **AnyLLM Provider**: according to AnyLLM provider naming (e.g. `gemini`, `openai`)

### Third row
- **Sketchfab API Key**: https://support.fab.com/s/article/Finding-your-API-Token

### Fourth row *(Meshy required)*
- **Meshy API Key**: https://www.meshy.ai/api
- **Meshy Model**: default `latest`

### Fifth row *(optional; for Hunyuan 3D Pro instead of Meshy)*
- **Tencent Cloud Secret ID** and **Tencent Cloud Secret Key**
  - EN: https://www.tencentcloud.com/document/product/1284/75281
  - CN: https://cloud.tencent.com/document/product/1804/123461
- **AI Platform**: choose which platform to use for 3D model generation: `Hunyuan3D`, `Meshy`, or `TRELLIS2`

### Sixth row *(optional; for local TRELLIS2)*
**Follow the instructions here to set up the server:** https://github.com/EngineeringAI-LAB/TRELLIS.2-Web-API
- **TRELLIS2 API Base**: default `http://127.0.0.1:7862/openapi/v1`
- **TRELLIS2 Max Concurrent**: default `1`. Keep this low unless your TRELLIS server has enough GPU memory for parallel jobs.
- **TRELLIS2 Texture Size**: default `4096`
- **TRELLIS2 Resolution**: default `1024`; supported values are `512`, `1024`, and `1536`
- **TRELLIS2 Decimation Target**: default `1000000`

To use this provider, start the TRELLIS.2 local API server from the TRELLIS.2 environment before running StoryBlender:

```bash
cd ../TRELLIS.2-Web-API
python trellis2_api_server.py --host 127.0.0.1 --port 7862
```

Then set:

```bash
STORYBLENDER_AI_PLATFORM=TRELLIS2
TRELLIS2_API_BASE=http://127.0.0.1:7862/openapi/v1
```

### Seventh row
- **Project Absolute Directory**: absolute path to your project directory to store all intermediate files for a story

---

## 🛠️ Workflow

Once the configuration is saved, follow the on-screen instructions in the web UI to complete the full storyboard generation and editing pipeline.

If you have questions, feel free to open a discussion:
- https://github.com/EngineeringAI-LAB/StoryBlender/discussions

---

## 📚 Citation

If you find this project useful in your research, please cite:

```bibtex
@misc{li2026storyblenderintershotconsistenteditable,
      title={StoryBlender: Inter-Shot Consistent and Editable 3D Storyboard with Spatial-temporal Dynamics}, 
      author={Bingliang Li and Zhenhong Sun and Jiaming Bian and Yuehao Wu and Yifu Wang and Hongdong Li and Yatao Bian and Huadong Mo and Daoyi Dong},
      year={2026},
      eprint={2604.03315},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.03315}, 
}
```
