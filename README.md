## Dataset

```C++
--dataset
  --train
  	---original
    ------.png
    ---target
    ------.png
    ---gt
    ------.png
  --test
    ---original
    ------.png
    ---target
    ------.png
    ---gt
    ------.png
  --val
    ---original
    ------.png
    ---target
    ------.png
    ---gt
    ------.png
```

```bash
cd Run
python train.py
```

## Requirements

- PyTorch == 1.13.0
- TorchVision == 0.14.0
- CUDA 11.8

```bash
sudo pip3 install -r requirements.txt
```
