import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree


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

        self.drop_path = drop_path

        self.norm1 = nn.LayerNorm(dim)

        self.attn = Attention(dim, num_heads,subkey1, qkv_bias, attn_drop, drop)

        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = dim * mlp_ratio

        self.fc1 = nn.Linear(dim, mlp_hidden, key=subkey2)
        self.fc2 = nn.Linear(mlp_hidden, dim, key=subkey3)
        self.mlp_drop = nn.Dropout(drop)

    def _drop_path(
            self,
            x: Array,
            key: PRNGKeyArray,
            inference: bool = False
    ):
        
        def _drop(x, key):
            key_prob = 1 - self.drop_path
            B = x.shape[0]
            shape = (B,) + (1,) * (x.ndim - 1)

            mask = jax.random.bernoulli(key, key_prob, shape)

            output = (x * mask) / key_prob
            
            return output
        
        return jax.lax.cond(
            self.drop_path > 0. or not inference,
            _drop,
            lambda x, k: x,
            x, key
        )

    def __call__(
            self,
            x: Array,
            state: nn._stateful.State,
            key: PRNGKeyArray | None = None,
            adapters: list[Adapter]| None = None,
            gates: list[Array]| None = None
    ):
        subkey1, subkey2, subkey3, subkey4, subkey5, key = jax.random.split(key, 6)
        x = x + self._drop_path(self.attn(self.norm1(x, state), subkey1), subkey2)
        residual = x

        x = self.norm2(x, state)
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.mlp_drop(x, subkey3)

        x = self.fc2(x)
        x = self.mlp_drop(x, subkey4)
        x = self._drop_path(x, subkey5)

        def adapter_loop(x,key):
            adapt_outputs = []
            for i, adapt in enumerate(adapters):
                subkey, key = jax.random.split(key)
                adapt_out = adapt(x, state, subkey, add_residual=False)
                adapt_outputs.append(adapt_out * gates[i])
            adapt_x = jnp.sum(adapt_outputs)
            x = x + adapt_x
            return x + residual
        
        x = jax.lax.cond(
            adapters is not None and gates is not None,
            adapter_loop,
            lambda x, k: x + residual,
            x, key
        )
        return key

class VisionTransformer(eqx.Module):
    num_classes: Int = eqx.field(static=True)
    patch_embed: nn.Conv2d
    cls_token: jax.Array
    dist_token: jax.Array
    pos_embed: jax.Array
    pos_drop: nn.Dropout
    blocks: list[Block]
    norm: nn.LayerNorm
    head: nn.Linear
    head_dist: nn.Linear
    embeddings: list[jax.Array]
    adapter_list: list
    gate_list: list

    def __init__(
            self,
            img_size: Int,
            patch_size: Int,
            in_chan: Int,
            num_classes: Int,
            key: PRNGKeyArray,
            embed_dim: Int = 1024,
            depth: Int = 12,
            num_heads: Int = 12,
            mlp_ratio: Float = 4.,
            qkv_bias: bool = True,
            drop_rate: Float = 0.,
            attn_drop_rate: Float = 0.,
            drop_path_rate: Float = 0.,
            tunning_config: dict | None = None,
    ):
        subkey1, subkey2 = jax.random.split(key, 2)
        self.num_classes = num_classes
        self.num_features = embed_dim

        self.patch_embed = nn.Conv2d(in_chan, embed_dim, kernel_size=patch_size, stride=patch_size, key=subkey1)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = jnp.zeros((1,1, embed_dim), dtype=jnp.float32)
        self.dist_token = jnp.zeros((1,1, embed_dim), dtype=jnp.float32)
        self.pos_embed = jnp.zeros((1, num_patches + 2, embed_dim), dtype=jnp.float32)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in jnp.linspace(0, drop_path_rate, depth)]
        key, *subkeys = jax.random.split(2, depth+1)

        self.blocks = [
            Block(
                embed_dim, num_heads, mlp_ratio, subkeys[i], qkv_bias, 
                drop_rate, attn_drop_rate, dpr[i]
            ) for i in range(depth)
        ]

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.head_dist = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.embeddings = [
            jnp.ones((1, tunning_config["vpt_num"], embed_dim), dtype=jnp.float32) for _ in range(depth)
        ]
        self.adapter_list = []
        self.gate_list = []