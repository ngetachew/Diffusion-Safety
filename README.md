Code for a project for CS592-TML.

# Overview

In this project, my goal is to modify the weights in the [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) model from Hugging Face to prevent it from accurately answering prompts that contain either toxic questions or seek to repeat copyrighted material. Using these prompts, I will create a set $F$ containing the hidden states of each of these prompts before the final layer of the diffusion model. Then, I will create an estimate low dimensional subspace of F called S with orthonormal basis U. I will construct a projection matrix $P=I - UU^T$ and multiply the final weight matrix with this, in order to effectively project out the subspace of undesired prompts.

### Llada architecture

LLaDAModelLM(
  (model): LLaDAModel(
    (transformer): ModuleDict(
      (wte): Embedding(126464, 4096)
      (emb_drop): Dropout(p=0.0, inplace=False)
      (ln_f): RMSLayerNorm()
      (blocks): ModuleList(
        (0-31): 32 x LLaDALlamaBlock(
          (dropout): Dropout(p=0.0, inplace=False)
          (act): SiLU()
          (attn_out): Linear(in_features=4096, out_features=4096, bias=False)
          (ff_out): Linear(in_features=12288, out_features=4096, bias=False)
          (rotary_emb): RotaryEmbedding()
          (attn_norm): RMSLayerNorm()
          (ff_norm): RMSLayerNorm()
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (ff_proj): Linear(in_features=4096, out_features=12288, bias=False)
          (up_proj): Linear(in_features=4096, out_features=12288, bias=False)
        )
      )
      (ff_out): Linear(in_features=4096, out_features=126464, bias=False)
    )
  )
)