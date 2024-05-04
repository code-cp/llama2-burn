use burn::backend::{
    ndarray::{NdArray, NdArrayDevice},
    Autodiff,
};
use burn::{
    config::Config,
    module::Module,
    nn::{attention, Gelu, LayerNorm},
    tensor::{backend::Backend, ElementConversion, Tensor},
};
use burn_tensor::{activation, Data, Int, Shape};
use std::io::{self, BufReader, ErrorKind, Read, Write};

use crate::state::RunState;
use crate::utils::*;

// pub type DefalutBackend = burn::backend::Autodiff<burn::backend::NdArray>;
pub type DefalutBackend = burn::backend::NdArray;
pub type DefaultDevice = burn_ndarray::NdArrayDevice;

#[derive(Debug, Config)]
pub struct StateConfig {
    pub dim: i32,
    pub hidden_dim: i32,
    pub n_layers: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub vocab_size: i32,
    pub seq_len: i32,
}

impl StateConfig {
    pub fn from_binary<R: Read>(input: &mut BufReader<R>) -> Self {
        let mut header = Vec::new();
        for _ in 0..7 {
            let x = read_i32(input).expect("config should have correct format");
            header.push(x);
        }

        Self {
            dim: header[0],
            hidden_dim: header[1],
            n_layers: header[2],
            n_heads: header[3],
            n_kv_heads: header[4],
            vocab_size: header[5],
            seq_len: header[6],
        }
    }

    pub fn init<B: Backend<Device = NdArrayDevice>>(&self, device: &B::Device) -> RunState {
        RunState {
            x: Tensor::zeros(Shape::new([self.dim as usize]), device),
            xb: Tensor::zeros(Shape::new([self.dim as usize]), device),
            xb2: Tensor::zeros(Shape::new([self.dim as usize]), device),
            hb: Tensor::zeros(Shape::new([self.hidden_dim as usize]), device),
            hb2: Tensor::zeros(Shape::new([self.hidden_dim as usize]), device),
            q: Tensor::zeros(Shape::new([self.dim as usize]), device),
            k: Tensor::zeros(Shape::new([self.dim as usize]), device),
            v: Tensor::zeros(Shape::new([self.dim as usize]), device),
            att: Tensor::zeros(Shape::new([self.seq_len as usize]), device),
            logits: Tensor::zeros(Shape::new([self.vocab_size as usize]), device),
            key_cache: Tensor::zeros(
                Shape::new([
                    self.n_layers as usize,
                    self.seq_len as usize,
                    self.dim as usize,
                ]),
                device,
            ),
            value_cache: Tensor::zeros(
                Shape::new([
                    self.n_layers as usize,
                    self.seq_len as usize,
                    self.dim as usize,
                ]),
                device,
            ),
        }
    }
}
