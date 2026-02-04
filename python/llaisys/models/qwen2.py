from typing import Sequence
from pathlib import Path
import json
import ctypes

import safetensors

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        # Load config.json
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)

        # Parse config
        hidden_size = config["hidden_size"]
        num_attention_heads = config["num_attention_heads"]
        num_key_value_heads = config["num_key_value_heads"]
        head_dim = hidden_size // num_attention_heads

        # Create meta structure
        self.meta = LlaisysQwen2Meta(
            dtype=DataType.BF16,
            nlayer=config["num_hidden_layers"],
            hs=hidden_size,
            nh=num_attention_heads,
            nkvh=num_key_value_heads,
            dh=head_dim,
            di=head_dim,  # kv head dim same as q head dim
            im=config["intermediate_size"],
            maxseq=config["max_position_embeddings"],
            voc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=config["rope_theta"],
            end_token=config["eos_token_id"],
        )

        # Create model
        self.device = device
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta),
            device,
            None,  # device_ids
            0,     # ndevice
        )

        # Get weights pointer
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        self.weights = weights_ptr.contents

        # Load weights from safetensors
        self._load_weights(model_path)

    def _load_weights(self, model_path: Path):
        """Load weights from safetensors files."""
        nlayer = self.meta.nlayer

        # Weight name mapping: safetensors name -> (weight_attr, layer_index or None)
        weight_map = {
            "model.embed_tokens.weight": ("in_embed", None),
            "lm_head.weight": ("out_embed", None),
            "model.norm.weight": ("out_norm_w", None),
        }

        # Add layer weights
        for i in range(nlayer):
            prefix = f"model.layers.{i}"
            weight_map.update({
                f"{prefix}.input_layernorm.weight": ("attn_norm_w", i),
                f"{prefix}.self_attn.q_proj.weight": ("attn_q_w", i),
                f"{prefix}.self_attn.q_proj.bias": ("attn_q_b", i),
                f"{prefix}.self_attn.k_proj.weight": ("attn_k_w", i),
                f"{prefix}.self_attn.k_proj.bias": ("attn_k_b", i),
                f"{prefix}.self_attn.v_proj.weight": ("attn_v_w", i),
                f"{prefix}.self_attn.v_proj.bias": ("attn_v_b", i),
                f"{prefix}.self_attn.o_proj.weight": ("attn_o_w", i),
                f"{prefix}.post_attention_layernorm.weight": ("mlp_norm_w", i),
                f"{prefix}.mlp.gate_proj.weight": ("mlp_gate_w", i),
                f"{prefix}.mlp.up_proj.weight": ("mlp_up_w", i),
                f"{prefix}.mlp.down_proj.weight": ("mlp_down_w", i),
            })

        # Load each safetensors file
        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    if name not in weight_map:
                        continue

                    attr_name, layer_idx = weight_map[name]
                    tensor_data = f.get_tensor(name)

                    # Get the target tensor handle
                    if layer_idx is None:
                        # Non-layer weight (embedding, norm)
                        tensor_handle = getattr(self.weights, attr_name)
                    else:
                        # Layer weight (array of tensors)
                        tensor_array = getattr(self.weights, attr_name)
                        tensor_handle = tensor_array[layer_idx]

                    # Transpose weight matrices if needed (linear layers use [out, in])
                    if "weight" in name and len(tensor_data.shape) == 2:
                        # HuggingFace format: [out_features, in_features]
                        # Our format: [in_features, out_features] for linear(x, w) = x @ w^T
                        # Actually the C++ linear does: out[j] = sum_k(in[k] * weight[j*in + k])
                        # which is weight[out, in], so no transpose needed
                        pass

                    # Load data to tensor
                    data_ptr = tensor_data.data_ptr()
                    LIB_LLAISYS.tensorLoad(tensor_handle, ctypes.c_void_p(data_ptr))

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """Generate tokens autoregressively.

        Args:
            inputs: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            top_k: Top-k sampling (1 = greedy)
            top_p: Top-p (nucleus) sampling threshold
            temperature: Sampling temperature

        Returns:
            List of generated token IDs (including input tokens)
        """
        if max_new_tokens is None:
            max_new_tokens = 128

        # Convert inputs to list
        tokens = list(inputs)
        end_token = self.meta.end_token

        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Prepare token array for C API
            token_array = (ctypes.c_int64 * len(tokens))(*tokens)

            # Run inference to get next token
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                token_array,
                len(tokens),
            )

            # Append to sequence
            tokens.append(next_token)

            # Check for end token
            if next_token == end_token:
                break

        return tokens

    def __del__(self):
        """Clean up model resources."""
        if hasattr(self, '_model') and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
