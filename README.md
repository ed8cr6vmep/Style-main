# Style-main

Code for *Style Transfer of Chinese Traditional Decorative Pattern based on Diffusion Model* (WDM).

## Setup

```bash
pip install -r requirements.txt
# optional: pip install git+https://github.com/openai/CLIP.git
```

## Train / infer / eval

```bash
python train.py --config configs/wdm.yaml --data data/CTDPD --out checkpoints
python infer.py --ckpt checkpoints/wdm.pt --content path/to/content.png --prompt "paper cutting of a peony"
python eval.py  --ckpt checkpoints/wdm.pt --data data/CTDPD
```

Hyper-parameters live in `configs/wdm.yaml` 