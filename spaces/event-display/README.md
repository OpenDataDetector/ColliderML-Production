---
title: ColliderML Event Display
emoji: 🔭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# ColliderML Event Display

Interactive 3D visualisation of single events from the
[ColliderML datasets](https://huggingface.co/datasets/CERN/ColliderML-Release-1).

Pick a physics process and an event ID to see the tracker hits, reconstructed
tracks, and truth particles inside the OpenDataDetector geometry.

## Local development

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

## Caching

The app pre-caches a small set of events per dataset on first load (via
`huggingface_hub`). Subsequent loads are instant.
