# EmbeddingGemma 2 Product Match Checker

Local product matching using Google's EmbeddingGemma 2 (open weights).
Supports reference image vs image or video. No API key needed.

## Setup
```bash
pip install -U "sentence-transformers[image,video]" transformers pillow opencv-python torch
python match.py --ref reference.jpg --comp video_or_image.mp4
```

First run will download the model (~1.5GB).
