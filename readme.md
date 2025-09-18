## ContentGuard: Multi-Modal Content Safety Chatbot

ContentGuard is a multi-modal content safety system that classifies both text and images for safety violations across multiple categories. Built using several machine learning models, it provides real-time content analysis with explainable reasoning through chain-of-thought processing. Users can leverage the chatbot to detect offensive content, NSFW material, hate speech, violence, and other harmful categories from both images and text. See below for a rundown of features:

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Text Safety | `patrickquick/BERTicelli` | Offensive content detection |
| Text Categorization | `facebook/bart-large-mnli` | Zero-shot safety category classification |
| Image Safety | `Falconsai/nsfw_image_detection` | NSFW content detection |
| Image Categorization | `openai/clip-vit-base-patch32` | Visual content categorization |
| OCR | `EasyOCR` | Text extraction from images |
| Reasoning | `llava-hf/llava-1.5-7b-hf` | Multi-modal chain-of-thought analysis |
