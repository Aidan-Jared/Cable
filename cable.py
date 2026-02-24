import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

from utils import Drop_Path, torch_to_equinox
from typing import Callable

from timm import create_model
import re

class Adapter(eqx.Module):
    down_proj: nn.Linear
    up_proj: nn.Linear
    dropout: nn.Dropout
    layer_norm: nn.LayerNorm
    scale: jax.Array
    adapter_layer_norm_option: str = eqx.field(static=True)

    def __init__(
            self,
            d_model: Int,
            down_size: Int,
            key: PRNGKeyArray,
            dropout: Float = .1,
            adapter_layer_norm_option: str = "in"
    ):
        subkey1, subkey2 = jax.random.split(key)

        self.adapter_layer_norm_option = adapter_layer_norm_option
        
        if adapter_layer_norm_option == "in" or adapter_layer_norm_option == "out":
            self.layer_norm = nn.LayerNorm(d_model)
        
        self.scale = jnp.ones(1, dtype=jnp.float32)

        self.down_proj = nn.Linear(d_model, down_size, key=subkey1)
        
        self.up_proj = nn.Linear(down_size, d_model, key=subkey2)

        self.dropout = nn.Dropout(dropout)

        self.down_proj = eqx.tree_at(
            lambda l: l.weight, 
            self.down_proj, 
            jnp.ones_like(self.down_proj.weight) * jnp.sqrt(5)
            )
        self.down_proj = eqx.tree_at(
            lambda l: l.bias, 
            self.down_proj, 
            jnp.zeros_like(self.down_proj.bias)
            )
        self.up_proj = eqx.tree_at(
            lambda l: l.weight, 
            self.up_proj, 
            jnp.zeros_like(self.down_proj.weight)
            )
        self.up_proj = eqx.tree_at(
            lambda l: l.bias, 
            self.up_proj, 
            jnp.zeros_like(self.down_proj.bias)
            )
    
    def __call__(
            self,
            x: Array,
            state: nn._stateful.State,
            key: PRNGKeyArray | None = None,
            add_residual: bool = True,
            residual: Array | None = None
    ):
        residual = jax.lax.cond(
            residual is None,
            lambda r, x: x,
            lambda r, x: r,
            residual, x
        )

        def forward(x, key):
            down = self.down_proj(x)
            down = jax.nn.relu(down)
            down = self.dropout(down, key=key)
            up = self.up_proj(down)
            return up * self.scale
        
        up = jax.lax.cond(
            self.adapter_layer_norm_option == "in",
            forward(self.layer_norm(x, state), key),
            self.layer_norm(forward(x, key), state)
        )

        output = jax.lax.cond(
            add_residual,
            lambda u: u + residual,
            lambda u: u,
            up
        )

        return output

class Attention(eqx.Module):
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    proj: nn.Linear
    attn_dropout: nn.Dropout
    proj_dropout: nn.Dropout
    num_heads: Int = eqx.field(static=True)
    head_dim: Int = eqx.field(static=True)
    scale: Float = eqx.field(static=True)

    def __init__(
            self, 
            dim: Int,
            num_heads: Int,
            key: PRNGKeyArray,
            qkv_bias: bool = False,
            attn_drop: Float = 0.,
            proj_drop: Float = 0.    
        ):

        subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 4)
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -.5

        self.q_proj = nn.Linear(dim, dim, use_bias=qkv_bias, key=subkey1)
        self.k_proj = nn.Linear(dim, dim, use_bias=qkv_bias, key=subkey2)
        self.v_proj = nn.Linear(dim, dim, use_bias=qkv_bias, key=subkey3)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, key=subkey4)
        self.proj_dropout = nn.Dropout(proj_drop)

    def _shape(
            self,
            array: Array,
            seq_len: Int,
            bsz: Int
    ):
        return array.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2)
    
    def __call__(
            self,
            x: Array,
            key: PRNGKeyArray| None = None
    ):
        subkey1, subkey2 = jax.random.split(key)
        B, N, C = x.shape
        
        q = self._shape(self.q_proj(x), N, B).reshape(B * self.num_heads, -1, self.head_dim)
        k = self._shape(self.k_proj(x), -1, B).reshape(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).reshape(B * self.num_heads, -1, self.head_dim)

        attn_weights = jnp.einsum("bdq, bdk -> bqk", q, k) * self.scale

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_probs = self.attn_dropout(attn_weights, subkey1)

        attn_output = jnp.einsum("bqk,bdv-> bdk", attn_probs, v)

        attn_output = attn_output.reshape(B, self.num_heads, N, self.head_dim).transpose(1,2).reshape(B,N,C)
        
        x = self.proj(attn_output)
        x = self.proj_dropout(x, subkey2)

        return x

class Block(eqx.Module):
    norm1: nn.LayerNorm
    attn: Attention
    norm2: nn.LayerNorm
    fc1: nn.Linear
    fc2: nn.Linear
    mlp_drop: nn.Dropout
    drop_path: Drop_Path

    def __init__(
            self,
            dim: Int,
            num_heads: Int,
            mlp_ratio: Int,
            key: PRNGKeyArray,
            qkv_bias: bool = False,
            drop: Float = 0.,
            attn_drop: Float = 0.,
            drop_path: Float = 0.
    ):
        subkey1, subkey2, subkey3 = jax.random.split(key,3)

        self.drop_path = Drop_Path(drop_path)

        self.norm1 = nn.LayerNorm(dim)

        self.attn = Attention(dim, num_heads,subkey1, qkv_bias, attn_drop, drop)

        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden, key=subkey2)
        self.fc2 = nn.Linear(mlp_hidden, dim, key=subkey3)
        self.mlp_drop = nn.Dropout(drop)

    def __call__(
            self,
            x: Array,
            state: nn._stateful.State,
            key: PRNGKeyArray | None = None,
            adapters: list[Adapter]| None = None,
            gates: list[Array]| None = None
    ):
        subkey1, subkey2, subkey3, subkey4, subkey5, key = jax.random.split(key, 6)
        x = x + self.drop_path(self.attn(self.norm1(x, state), subkey1), subkey2)
        residual = x

        x = self.norm2(x, state)
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.mlp_drop(x, subkey3)

        x = self.fc2(x)
        x = self.mlp_drop(x, subkey4)
        x = self.drop_path(x, subkey5)

        def adapter_loop(x,key):
            adapt_outputs = []
            def _gated(x, key):
                for i, adapt in enumerate(adapters):
                    subkey, key = jax.random.split(key)
                    adapt_out = adapt(x, state, subkey, add_residual=False)
                    adapt_outputs.append(adapt_out * gates[i])
                return jnp.sum(adapt_outputs)
            adapt_x = jax.lax.cond(
                gates != [],
                _gated,
                lambda x, k: adapters[0](x, state, k, add_residual=False),
                x, key
            )
            x = x + adapt_x
            return x + residual
        
        x = jax.lax.cond(
            adapters is not None,
            adapter_loop,
            lambda x, k: x + residual,
            x, key
        )
        return key

class PatchEmbed(eqx.Module):
    proj: nn.Conv2d
    norm: nn.LayerNorm | nn.Identity
    patch_size: Int
    img_size: Int
    num_patches: Int
    embed_dim: Int

    def __init__(
            self,
            img_size: Int = 224,
            patch_size: Int = 16,
            in_channels: Int = 3,
            embed_dim: Int = 768,
            norm_layer: bool = False,
            *,
            key: PRNGKeyArray
        ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size)**2
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            key=key
        )

        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()

    def __call__(
            self,
            x: Array
    ):
        x = self.proj(x)
        x = x.reshape(self.embed_dim, -1).T

        x = jax.vmap(self.norm)(x)
        return x

class VisionTransformer(eqx.Module):
    num_classes: Int = eqx.field(static=True)
    embed_dim: Int = eqx.field(static=True)
    patch_embed: PatchEmbed
    cls_token: jax.Array
    dist_token: jax.Array
    pos_embed: jax.Array
    pos_drop: nn.Dropout
    blocks: list[Block]
    norm: nn.LayerNorm
    head: nn.Linear
    # head_dist: nn.Linear
    # embeddings: list[jax.Array]
    adapter_list: list
    adapter_gates: list

    def __init__(
            self,
            img_size: Int = 224,
            patch_size: Int = 16,
            in_chan: Int = 3,
            num_classes: Int = 1000,
            embed_dim: Int = 1024,
            depth: Int = 12,
            num_heads: Int = 12,
            mlp_ratio: Float = 4.,
            qkv_bias: bool = True,
            drop_rate: Float = 0.,
            attn_drop_rate: Float = 0.,
            drop_path_rate: Float = 0.,
            tunning_config: dict | None = None,
            *,
            key: PRNGKeyArray,
    ):
        subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 4)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chan, embed_dim, key=subkey1)

        self.cls_token = jnp.zeros((1,1, embed_dim), dtype=jnp.float32)
        self.dist_token = jnp.zeros((1,1, embed_dim), dtype=jnp.float32)
        self.pos_embed = jnp.zeros((1, self.patch_embed.num_patches + 2, embed_dim), dtype=jnp.float32)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in jnp.linspace(0, drop_path_rate, depth)]
        key, *subkeys = jax.random.split(subkey2, depth+1)

        self.blocks = [
            Block(
                embed_dim, num_heads, mlp_ratio, subkeys[i], qkv_bias, 
                drop_rate, attn_drop_rate, dpr[i]
            ) for i in range(depth)
        ]

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes, key=subkey3) if num_classes > 0 else nn.Identity()

        # self.head_dist = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # self.embeddings = [
        #     jnp.ones((1, tunning_config["vpt_num"], embed_dim), dtype=jnp.float32) for _ in range(depth)
        # ]
        self.adapter_list = []
        self.adapter_gates = []
        self.get_new_adapter(subkey4)
    
    def set_adapter_gates(
            self,
            fisher_traces: Array,
            temp: Float = 3.
    ):
        fisher = jnp.array(fisher_traces, dtype=jnp.float32)
        self.adapter_gates = jax.nn.softmax(fisher / temp, axis=0)

    def compute_forgetting_score_adapters(
            self,
            forgetting_traces,
            inputs: Array,
            targets: Array,
            loss_fn: Callable,
            lr: Float = 1e-3
    ):
        targets = jnp.array(targets, dtype=jnp.int16)
        theta_list = []
        for adapter in self.adapter_list:
            theta_list.append(jax.tree_util.tree_leaves(eqx.filter(adapter, eqx.is_inexact_array)))
        
        grads = eqx.filter_grad(loss_fn)(self(inputs), targets)

        theta_t_list = []
        for idx, adapter in enumerate(self.adapter_list):
            params, _ = eqx.partition(adapter, eqx.is_array)
            adapter_grad = grads.adapter_list[idx]
            up_params = jax.tree.map(
                lambda p, g: p - lr * g,
                params,
                adapter_grad
            )
            theta_t_list.append(jax.tree_util.tree_leaves(eqx.filter(up_params, eqx.is_inexact_array)))
        
        scores = []
        for F, theta, theta_t in zip(forgetting_traces, theta_list, theta_t_list):
            score = 0.
            for f, p, p_t in zip(F, theta, theta_t):
                score += jnp.sum(f * (p - p_t)**2).item()
            scores.append(score)
        return self.zscore_normalize(scores)
    
    def zscore_normalize(self, scores: list):
        scores = jnp.array(scores)
        mean = jnp.mean(scores)
        std = jnp.mean(scores)

        return jax.lax.cond(
            std > 0,
            lambda s, m, st: ((s - m) / st).tolist(),
            lambda s, m, st: [0.0 for _ in s],
            scores, mean, std
        )
    
    def sum_fisher_traces(self, traces):
        sum_traces = []
        for adapter_traces in traces:
            total = 0.
            for param_trace in adapter_traces:
                total += jnp.sum(param_trace).item()
            sum_traces.append(total)
        return sum_traces
    
    def compute_fisher_trace(
            self,
            dataloader: Callable,
            loss_fn: Callable,
            num_batches: Int = 10
    ):
        traces = jax.tree.map(
            lambda p: jnp.zeros_like(p),
            self.adapter_list    
        )

        for _ in range(num_batches):
            inputs, _, targets = dataloader.__iter__()
            grads = eqx.filter_grad(loss_fn)(self(inputs), targets)

            grads = eqx.tree_at(lambda g: g.adapter_list, grads)
            traces = jax.tree.map(
                lambda t, g: t + g**2,
                traces,
                grads
            )
        
        traces = jax.tree.map(
            lambda t: t / num_batches,
            traces
        )

        self.set_adapter_gates(traces) # needs to be like trace[0].sum().item() but I need to see it first
        return traces
    
    def get_new_adapter(self, key):
        for i in range(len(self.blocks)):
            subkey, key = jax.random.split(key)
            adapter = Adapter(self.embed_dim, self.embed_dim // 2, key=subkey)
            self.adapter_list.append(adapter)
    
    def __call__(
            self,
            x: Array,
            state,
            key: PRNGKeyArray | None = None
    ):
        
        subkey, key = jax.random.split(key)
        B = x.shape[0]
        x = self.patch_embed(x)

        x = jnp.concat((self.cls_token, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x, subkey)

        for idx, blk in enumerate(self.blocks):
            subkey, key = jax.random.split(key)
            # x = jnp.concat(self.embeddings[idx], x, dim=1)

            x = blk(x, state, key, self.adapter_list, self.adapter_gates)
        x = self.norm(x, state)
        outcome = x[:,0]

        return outcome
    
if __name__ == "__main__":
    seed = 42
    key = jax.random.PRNGKey(seed)
    subkey1, subkey2 = jax.random.split(key)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, key = subkey1)

    checkpoint_model = create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    
    new_model = torch_to_equinox(model, state_dict, 768)
    print("hi")