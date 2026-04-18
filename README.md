# Fine tuning Gemma-4-31b-it procedure:
Local SFT

### Requirements Hardware & Software
- Computer with Nvidia GPU VRAM 32Gb or greater
- Linux Ubuntu OS with latest updates
- Nvidia GPU drivers
- Unsloth Studio
- Training dataset.jsonl (prepared according https://github.com/RH-Electronics/SFT_Dataset_Guide )
- PyCharm for editing python scripts (not mandatory)

### Setting GPU power limit
Unless you have a liquid cooled GPU, I suggest to set 70%-80% GPU power limit. It results in slower speed for SFT, but keeps your GPU in safe zone.

Open Terminal:
```python
nvidia-smi -q -d POWER
```

Find: Min Power Limit, in my case 400 

Apply this wattage:
```python
sudo nvidia-smi -pl 400
sudo nvidia-smi -pm 1
watch -n 1 nvidia-smi
```

### Open Unsloth Studio
Select **unsloth/gemma-4-31B-it** model, upload your dataset.jsonl
Apply SFT setting from the attached yaml or insert settings manually (my comments in (...)):

```python
  max_seq_length: 2048 (or according to your dataset plan, VRAM limitation)
  num_epochs: 3 (depend on dataset size, if you have 10K pairs with homogeneous style then maybe 1-2 epoch maybe enough)
  learning_rate: 0.0001 (trial and error method to find optimal for your dataset, start with 0.0001)
  batch_size: 1 (VRAM limitation)
  gradient_accumulation_steps: 32 (how often to update model weights during SFT, 32 found as a sweet point)
  warmup_steps: 7 ( warmup_steps  = (0.1* pairs * epoch)/(gradient_accumulation_steps * batch_size)
  max_steps: 0
  save_steps: 30
  eval_steps: 0
  weight_decay: 0.01
  random_seed: 3407
  packing: false
  train_on_completions: true
  gradient_checkpointing: unsloth
  optim: adamw_8bit
  lr_scheduler_type: cosine
lora:
  lora_r: 32 (larger values inject Lora adapter deeper, but may increase VRAM usage)
  lora_alpha: 64 (lora_r * 2)
  lora_dropout: 0.05 (for small dataset below 1000 pairs, if dataset is large like 10K then can be set to zero)
  target_modules:
    - all-linear
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  use_rslora: true
  finetune_vision_layers: false (do not train visual layers, set to false manually in yaml)
```

### Evaluating SFT statistics:
**Training Loss**: the graph should decrease its values down to 1.1-1.4  Final value below 1.0 - model overfit, it becomes dataset parroting -> reduce learning steps. Final value above 2.5 may tell that model is still not learned well dataset style -> add learning steps.
If you don't want to add learning steps, try to experimenting with lora_r value

**Gradient Norm**: better if the graph is stable flat or has slow decreasing. Sometimes with batch_size=1 you can see gradient explosure peaks, but if it returns to stable flat graph at the next evaluation step then it acceptable. If you see chaotic jumps it tells model is not stable.

<img width="1590" height="1210" alt="Screenshot from 2026-04-18 15-28-07" src="https://github.com/user-attachments/assets/af0bb74d-7fd4-41e3-97d0-762392b75ff0" />

### Exporting
Make frist chat test inside the unsloth studio chat. For Gemma-4 you can try:
```python
Temperature = 0.8 - 1.0
Top_P = 0.95
Top_K = 64
Min_P = 0.05
Rep.Penalty = 1.15
Presense Penalty = 0.3
```

If everything is fine open Export tab:
- Merge Model weights
- Export GGUF q8_0 or q5_K_M or q4_K_M

### Inference local Software
- LMStudio (gguf)
- WebUI (ollama gguf)
- Oobabooga (merged Safetensors or gguf)
