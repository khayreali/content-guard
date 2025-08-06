## ContentGuard: Multi-Modal Content Safety Chatbot
### About

ContentGuard is a multi-modal content safety system that classifies both text and images for safety violations across multiple categories. Built using several machine learning models, it provides real-time content analysis with explainable reasoning through chain-of-thought processing. Users can leverage the chatbot to detect offensive content, NSFW material, hate speech, violence, and other harmful categories from both images and text. See below for a rundown of features:

- **Multi-Modal Analysis**: Processes both text and image content.
- **Real-Time Classification**: Instant safety assessments.
- **Explainable AI**: Chain-of-thought reasoning provides transparent decision-making process
- **Multiple Safety Categories**: Detects Violence, Sexually Explicit content, Identity Hate, Drugs, Weapons, and Self-Harm
- **OCR Integration**: Extracts and analyzes text within images.
- **Interactive Web Interface**: Built with Streamlit.
- **Optimized Performance**: Model caching and device-aware acceleration (MPS/CPU)

### Core Components

**Text Classification Pipeline**:
- Primary classifier using BERTicelli model for offensive content detection
- Zero-shot classification with BART-large-mnli for category assignment
- Handles both direct text input and OCR-extracted text from images

**Image Classification Pipeline**:
- NSFW detection using specialized Falconsai model
- CLIP-based zero-shot categorization for safety violations
- EasyOCR integration for text extraction from images

**Chain-of-Thought Reasoning**:
- LLaVA 1.5 7B model provides step-by-step analysis
- Validates classification results with human-readable explanations
- Cross-modal reasoning for images containing text

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Text Safety | `patrickquick/BERTicelli` | Offensive content detection |
| Text Categorization | `facebook/bart-large-mnli` | Zero-shot safety category classification |
| Image Safety | `Falconsai/nsfw_image_detection` | NSFW content detection |
| Image Categorization | `openai/clip-vit-base-patch32` | Visual content categorization |
| OCR | `EasyOCR` | Text extraction from images |
| Reasoning | `llava-hf/llava-1.5-7b-hf` | Multi-modal chain-of-thought analysis |