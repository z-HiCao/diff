---
license: mit
---

# 🍰 Tiny AutoEncoder for Stable Diffusion

[TAESD](https://github.com/madebyollin/taesd) is very tiny autoencoder which uses the same "latent API" as Stable Diffusion's VAE.
TAESD is useful for [real-time previewing](https://twitter.com/madebyollin/status/1679356448655163394) of the SD generation process.

This repo contains `.safetensors` versions of the TAESD weights.

For SDXL, use [TAESDXL](https://huggingface.co/madebyollin/taesdxl/) instead (the SD and SDXL VAEs are [incompatible](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/discussions/6#64b8a9c13707b7d603c6ac16)).

## Using in 🧨 diffusers

```python
import torch
from diffusers import DiffusionPipeline, AutoencoderTiny

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16
)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "slice of delicious New York-style cheesecake topped with berries, mint, chocolate crumble"
image = pipe(prompt, num_inference_steps=50, generator=torch.Generator("cpu").manual_seed(0x7A35D)).images[0]
image.save("cheesecake.png")
```

![image/png](https://cdn-uploads.huggingface.co/production/uploads/630447d40547362a22a969a2/m4pVdlJ25U774v04Tsgzu.png)