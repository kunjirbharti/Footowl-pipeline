# FotoOwl AI — Image-to-Video Multiagent Pipeline

An AI-powered multi-agent pipeline that converts a collection of images into a polished video reel using **LangGraph**, **RAG**, **Google Gemini**, and **Remotion**.

Built as part of the FotoOwl AI Engineering Internship assignment to demonstrate expertise in multi-agent orchestration, retrieval-augmented generation, structured outputs, and automated video generation.

---

##  Features

-  Intent parsing from natural language prompts
-  Intelligent image selection and analysis
-  AI-generated storyboard creation
-  Retrieval-Augmented Generation (RAG) for style guidance
-  LangGraph multi-agent workflow
-  Automatic Remotion React video generation
-  Compiler validation with retry mechanism
-  Pipeline state tracking
-  Automated video rendering

---

##  Architecture

```
User Prompt
      │
      ▼
Intent Parser Agent
      │
      ▼
Image Analysis Agent
      │
      ▼
RAG Retriever
      │
      ▼
Storyboard Generator
      │
      ▼
Remotion Script Generator
      │
      ▼
Compiler & Auto Fix Agent
      │
      ▼
Renderer
      │
      ▼
Final Video
```

---

## Multi-Agent Workflow

### 1. Intent Parser
Extracts:
- Video type
- Mood
- Theme
- Caption style
- Color palette
- Transition style

---

### 2. Image Analyzer

Analyzes uploaded images and selects the most relevant ones based on the user's prompt.

---

### 3. Storyboard Writer

Uses RAG to retrieve cinematic style guides and generates:

- Scene sequence
- Duration
- Captions
- Camera movements
- Visual effects

---

### 4. Script Generator

Creates a complete **Remotion React component** for rendering the slideshow.

---

### 5. Compiler & Fixer

Validates generated code and retries automatically if compilation fails.

---

### 6. Renderer

Generates the final slideshow video using Remotion.

---

## Project Structure

```
fotoowl-pipeline/
│
├── agents.py
├── graph.py
├── rag.py
├── schemas.py
├── run_pipeline.py
├── run_demo.py
├── prompts.py
├── utils.py
│
├── remotion-app/
│
├── test_images/
│
├── sample_output/
│   ├── wedding/
│   └── birthday/
│
├── knowledge_base/
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

- Python
- LangGraph
- LangChain
- Google Gemini API
- ChromaDB
- RAG
- FAISS / Vector Search
- Pydantic
- Remotion
- React
- Node.js

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/fotoowl-pipeline.git

cd fotoowl-pipeline
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file.

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Generate an API key from:

https://aistudio.google.com/app/apikey

---

##  Run

```bash
python run_demo.py
```

---

##  Output

The pipeline generates:


sample_output/

├── storyboard.json
├── Slideshow.tsx
├── pipeline_state.json


and renders


remotion-app/out/video.mp4


##  Skills Demonstrated

- Multi-Agent AI Systems
- LangGraph
- Retrieval-Augmented Generation (RAG)
- Google Gemini API
- Structured Outputs
- Prompt Engineering
- Vector Retrieval
- Automated Code Generation
- React Code Synthesis
- Video Generation
- Pipeline Orchestration
- Error Recovery
- Python
