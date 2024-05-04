use burn::{
    config::Config,
    module::Module,
    nn::{attention, Gelu, LayerNorm},
    tensor::{backend::Backend, ElementConversion, Tensor},
};
use burn_tensor::{activation, Data, Int, Shape};

use crate::config::DefalutBackend as B;

#[derive(Debug, Clone)]
pub struct RunState {
    /// activation at current t (dim,)
    pub x: Tensor<B, 1>,
    /// same but inside residual branch (dim,)
    pub xb: Tensor<B, 1>,
    /// extra buffer for convenience (dim,)
    pub xb2: Tensor<B, 1>,
    /// buffer for hidden dim in ffn (hidden_dim,)
    pub hb: Tensor<B, 1>,
    /// buffer for hidden dim in ffn (hidden_dim,)
    pub hb2: Tensor<B, 1>,
    /// query (dim,)
    pub q: Tensor<B, 1>,
    /// key (dim,)
    pub k: Tensor<B, 1>,
    /// value (dim,)
    pub v: Tensor<B, 1>,
    /// buffer for scores/attn values (seq_len,)
    pub att: Tensor<B, 1>,
    /// (output_logits,)
    pub logits: Tensor<B, 1>,
    /// (layer, seq_len, dim)
    pub key_cache: Tensor<B, 3>,
    /// (layer, seq_len, dim)
    pub value_cache: Tensor<B, 3>,
}
