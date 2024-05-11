use burn::backend::{
    ndarray::{NdArray, NdArrayDevice},
    Autodiff,
};
use burn::{
    config::Config,
    module::Module,
    nn::{attention, Gelu, LayerNorm, RmsNorm, RmsNormConfig},
    tensor::{backend::Backend, ElementConversion, Tensor},
};
use burn_ndarray::NdArrayTensor;
use burn_tensor::{activation, Data, Int, Shape};
use ndarray::prelude::*;
use std::fs::{read, File};
use std::io::{self, BufReader, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use crate::config::{DefaultDevice, StateConfig};
use crate::utils::*;
use crate::{config::DefalutBackend as B, state::RunState};

#[derive(Debug, Config, Shrinkwrap)]
pub struct ModelConfig {
    #[shrinkwrap(main_field)]
    pub state_config: StateConfig,
}

impl ModelConfig {
    pub fn init_zero(&self, device: &DefaultDevice) -> Model {
        let token_embedding = Tensor::ones(
            Shape::new([self.vocab_size as usize, self.dim as usize]),
            device,
        );

        let rms_att_weight = Tensor::ones(
            Shape::new([self.n_layers as usize, self.dim as usize]),
            device,
        );

        let rms_ffn_weight = Tensor::ones(
            Shape::new([self.n_layers as usize, self.dim as usize]),
            device,
        );

        let wq = Tensor::ones(
            Shape::new([self.n_layers as usize, self.dim as usize, self.dim as usize]),
            device,
        );

        let wk = Tensor::ones(
            Shape::new([self.n_layers as usize, self.dim as usize, self.dim as usize]),
            device,
        );

        let wv = Tensor::ones(
            Shape::new([self.n_layers as usize, self.dim as usize, self.dim as usize]),
            device,
        );

        let wo = Tensor::ones(
            Shape::new([self.n_layers as usize, self.dim as usize, self.dim as usize]),
            device,
        );

        let w1 = Tensor::ones(
            Shape::new([
                self.n_layers as usize,
                self.hidden_dim as usize,
                self.dim as usize,
            ]),
            device,
        );

        let w2 = Tensor::ones(
            Shape::new([
                self.n_layers as usize,
                self.dim as usize,
                self.hidden_dim as usize,
            ]),
            device,
        );

        let w3 = Tensor::ones(
            Shape::new([
                self.n_layers as usize,
                self.hidden_dim as usize,
                self.dim as usize,
            ]),
            device,
        );

        let rms_final_weight = Tensor::ones(Shape::new([self.dim as usize]), device);

        let weights = ModelWeights {
            token_embedding,
            rms_att_weight,
            rms_ffn_weight,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_final_weight,
        };

        let run_state = self.init::<B>(device);

        let n_heads = self.n_heads;
        let head_size = self.dim / self.n_heads;
        let n_layers = self.n_layers;
        let rms_norm = RmsNormConfig::new(self.dim as usize)
            .with_epsilon(1e-5)
            .init::<B>(&device);

        Model {
            weights,
            run_state,
            n_heads,
            head_size,
            n_layers,
            rms_norm,
        }
    }

    /// NOTE, must load the config from model file first, then load weights
    pub fn init_from_weights<R: Read>(
        &self,
        device: &DefaultDevice,
        input: &mut BufReader<R>,
    ) -> Model {
        let mut read_weights = |size: (usize, usize, usize)| {
            let num = size.0 * size.1 * size.2;
            let mut bytes = Vec::new();
            for _ in 0..num {
                let byte = read_f32(input).expect("weights should have correct format");
                bytes.push(byte);
            }
            Array::from_shape_vec(size, bytes).expect("should be able to convert to ndarray")
        };

        let s2 = s![.., .., 0];
        let token_embedding_arr =
            read_weights((self.vocab_size as usize, self.dim as usize, 1)).slice_move(s2);
        let token_embedding = NdArrayTensor::<f32, 2>::new(token_embedding_arr.into_dyn().into());
        let token_embedding = Tensor::from_primitive(token_embedding);

        let rms_att_weight_arr =
            read_weights((self.n_layers as usize, self.dim as usize, 1)).slice_move(s2);
        let rms_att_weight = NdArrayTensor::<f32, 2>::new(rms_att_weight_arr.into_dyn().into());
        let rms_att_weight = Tensor::from_primitive(rms_att_weight);

        let wq_arr = read_weights((self.n_layers as usize, self.dim as usize, self.dim as usize));
        let wq_arr = NdArrayTensor::<f32, 3>::new(wq_arr.into_dyn().into());
        let wq = Tensor::from_primitive(wq_arr);

        let wk_arr = read_weights((self.n_layers as usize, self.dim as usize, self.dim as usize));
        let wk_arr = NdArrayTensor::<f32, 3>::new(wk_arr.into_dyn().into());
        let wk = Tensor::from_primitive(wk_arr);

        let wv_arr = read_weights((self.n_layers as usize, self.dim as usize, self.dim as usize));
        let wv_arr = NdArrayTensor::<f32, 3>::new(wv_arr.into_dyn().into());
        let wv = Tensor::from_primitive(wv_arr);

        let wo_arr = read_weights((self.n_layers as usize, self.dim as usize, self.dim as usize));
        let wo_arr = NdArrayTensor::<f32, 3>::new(wo_arr.into_dyn().into());
        let wo = Tensor::from_primitive(wo_arr);

        let rms_ffn_weight_arr =
            read_weights((self.n_layers as usize, self.dim as usize, 1)).slice_move(s2);
        let rms_ffn_weight_arr = NdArrayTensor::<f32, 2>::new(rms_ffn_weight_arr.into_dyn().into());
        let rms_ffn_weight = Tensor::from_primitive(rms_ffn_weight_arr);

        let w1_arr = read_weights((
            self.n_layers as usize,
            self.hidden_dim as usize,
            self.dim as usize,
        ));
        let w1_arr = NdArrayTensor::<f32, 3>::new(w1_arr.into_dyn().into());
        let w1 = Tensor::from_primitive(w1_arr);

        let w2_arr = read_weights((
            self.n_layers as usize,
            self.dim as usize,
            self.hidden_dim as usize,
        ));
        let w2_arr = NdArrayTensor::<f32, 3>::new(w2_arr.into_dyn().into());
        let w2 = Tensor::from_primitive(w2_arr);

        let w3_arr = read_weights((
            self.n_layers as usize,
            self.hidden_dim as usize,
            self.dim as usize,
        ));
        let w3_arr = NdArrayTensor::<f32, 3>::new(w3_arr.into_dyn().into());
        let w3 = Tensor::from_primitive(w3_arr);

        let rms_final_weight_arr = read_weights((self.dim as usize, 1, 1)).slice_move(s![.., 0, 0]);
        let rms_final_weight_arr =
            NdArrayTensor::<f32, 1>::new(rms_final_weight_arr.into_dyn().into());
        let rms_final_weight = Tensor::from_primitive(rms_final_weight_arr);

        let weights = ModelWeights {
            token_embedding,
            rms_att_weight,
            rms_ffn_weight,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_final_weight,
        };

        let run_state = self.init::<B>(device);

        let n_heads = self.n_heads;
        let head_size = self.dim / self.n_heads;
        let n_layers = self.n_layers;
        let rms_norm = RmsNormConfig::new(self.dim as usize)
            .with_epsilon(1e-5)
            .init::<B>(&device);

        Model {
            weights,
            run_state,
            n_heads,
            head_size,
            n_layers,
            rms_norm,
        }
    }
}

#[derive(Debug, Clone, Module)]
pub struct ModelWeights {
    /// (vocab_size, dim)
    pub token_embedding: Tensor<B, 2>,
    /// rms norms
    /// (layer, dim)
    pub rms_att_weight: Tensor<B, 2>,
    pub rms_ffn_weight: Tensor<B, 2>,
    /// (layer, dim, dim)
    pub wq: Tensor<B, 3>,
    pub wk: Tensor<B, 3>,
    pub wv: Tensor<B, 3>,
    pub wo: Tensor<B, 3>,
    /// ffn
    /// (layer, hidden_dim, dim)
    pub w1: Tensor<B, 3>,
    /// (layer, dim, hidden_dim)
    pub w2: Tensor<B, 3>,
    /// (layer, hidden_dim, dim)
    pub w3: Tensor<B, 3>,
    /// rms norm
    /// (dim,)
    pub rms_final_weight: Tensor<B, 1>,
}

#[derive(Debug, Clone, Module, Shrinkwrap)]
pub struct Model {
    #[shrinkwrap(main_field)]
    pub weights: ModelWeights,
    pub run_state: RunState,
    pub n_heads: i32,
    /// default is 48
    pub head_size: i32,
    pub n_layers: i32,
    pub rms_norm: RmsNorm<B>,
}

impl Model {
    pub fn forward(&mut self, token_id: usize, position: i32) {
        let device = <NdArray as Backend>::Device::default();

        let token_embedding_index = Tensor::from_ints([token_id as i32], &device);
        self.run_state.x = self
            .token_embedding
            .clone()
            .select(0, token_embedding_index)
            .squeeze(0);

        let model_dim = self.run_state.x.dims()[0];

        for layer_id in 0..self.n_layers {
            let layer_index = Tensor::from_ints([layer_id], &device);
            let att_weight: Tensor<B, 1> = self
                .rms_att_weight
                .clone()
                .select(0, layer_index.clone())
                .squeeze(0);
            self.run_state.xb = self.rms_norm.forward(self.run_state.x.clone()) * att_weight;

            // attention matmul
            // (1x288) x (288x288) = (1x288)
            self.run_state.q = self
                .run_state
                .xb
                .clone()
                .unsqueeze_dim::<2>(0)
                .matmul(
                    self.wq
                        .clone()
                        .select(0, layer_index.clone())
                        .squeeze(0)
                        .transpose(),
                )
                .squeeze(0);

            self.run_state.k = self
                .run_state
                .xb
                .clone()
                .unsqueeze_dim::<2>(0)
                .matmul(
                    self.wk
                        .clone()
                        .select(0, layer_index.clone())
                        .squeeze(0)
                        .transpose(),
                )
                .squeeze(0);

            self.run_state.v = self
                .run_state
                .xb
                .clone()
                .unsqueeze_dim::<2>(0)
                .matmul(
                    self.wv
                        .clone()
                        .select(0, layer_index.clone())
                        .squeeze(0)
                        .transpose(),
                )
                .squeeze(0);

            Model::rope(
                &mut self.run_state.q,
                self.n_heads as usize,
                self.head_size as usize,
                position as usize,
                &device,
            );

            Model::rope(
                &mut self.run_state.k,
                self.n_heads as usize,
                self.head_size as usize,
                position as usize,
                &device,
            );

            // save key and value to cache
            self.run_state.key_cache = self.run_state.key_cache.clone().slice_assign(
                [
                    layer_id as usize..(layer_id + 1) as usize,
                    position as usize..(position + 1) as usize,
                    0..model_dim,
                ],
                self.run_state.k.clone().unsqueeze_dims(&[0, 0]),
            );

            self.run_state.value_cache = self.run_state.value_cache.clone().slice_assign(
                [
                    layer_id as usize..(layer_id + 1) as usize,
                    position as usize..(position + 1) as usize,
                    0..model_dim,
                ],
                self.run_state.v.clone().unsqueeze_dims(&[0, 0]),
            );

            // multi-head attention
            self.attention(layer_id as usize, position as usize);

            // final attention matmul
            // xb2 is xb @ wo
            self.run_state.xb2 = self
                .wo
                .clone()
                .slice([
                    layer_id as usize..(layer_id + 1) as usize,
                    0..model_dim as usize,
                    0..model_dim as usize,
                ])
                .squeeze::<2>(0)
                .matmul(self.run_state.xb.clone().unsqueeze_dim(1))
                .squeeze(1);

            // residual connection
            self.run_state.x = self.run_state.x.clone() + self.run_state.xb2.clone();

            // ffn rmsnorm
            let rms_ffn_weight: Tensor<B, 1> = self
                .rms_ffn_weight
                .clone()
                .select(0, layer_index.clone())
                .squeeze(0);
            self.run_state.xb = self.rms_norm.forward(self.run_state.x.clone()) * rms_ffn_weight;

            self.feedforward(layer_id as usize);
        }

        // final rms norm
        self.run_state.x =
            self.rms_norm.forward(self.run_state.x.clone()) * self.rms_final_weight.clone();

        // logits
        self.run_state.logits = self
            .token_embedding
            .clone()
            .matmul(self.run_state.x.clone().unsqueeze_dim(1))
            .squeeze(1);
    }

    /// self.w2(F.silu(self.w1(x)) * self.w3(x))
    pub fn feedforward(&mut self, layer_id: usize) {
        let hidden_dim = self.w1.dims()[1];
        let model_dim = self.w1.dims()[2];

        self.run_state.hb = self
            .w1
            .clone()
            .slice([layer_id..layer_id + 1, 0..hidden_dim, 0..model_dim])
            .squeeze::<2>(0)
            .matmul(self.run_state.xb.clone().unsqueeze_dim(1))
            .squeeze(1);

        self.run_state.hb = activation::silu(self.run_state.hb.clone());

        self.run_state.hb2 = self
            .w3
            .clone()
            .slice([layer_id..layer_id + 1, 0..hidden_dim, 0..model_dim])
            .squeeze::<2>(0)
            .matmul(self.run_state.xb.clone().unsqueeze_dim(1))
            .squeeze(1);

        self.run_state.hb = self.run_state.hb.clone() * self.run_state.hb2.clone();

        self.run_state.xb = self
            .w2
            .clone()
            .slice([layer_id..layer_id + 1, 0..model_dim, 0..hidden_dim])
            .squeeze::<2>(0)
            .matmul(self.run_state.hb.clone().unsqueeze_dim(1))
            .squeeze(1);

        // residual connection
        self.run_state.x = self.run_state.x.clone() + self.run_state.xb.clone();
    }

    pub fn attention(&mut self, layer_id: usize, position: usize) {
        for head_index in 0..self.n_heads {
            let q_head = self
                .run_state
                .q
                .clone()
                .slice([(head_index * self.head_size) as usize
                    ..((head_index + 1) * self.head_size) as usize]);

            let prev_key: Tensor<B, 2> = self
                .run_state
                .key_cache
                .clone()
                .slice([
                    layer_id..layer_id + 1,
                    0..position + 1,
                    (head_index * self.head_size) as usize
                        ..((head_index + 1) * self.head_size) as usize,
                ])
                .squeeze::<2>(0);

            // softmax(Q K^T / sqrt(d))
            let attn_logits = q_head
                .clone()
                .unsqueeze()
                .matmul(prev_key.clone().transpose())
                / (self.head_size as f32).sqrt();

            let scores = activation::softmax(attn_logits.clone(), 1).squeeze(0);

            self.run_state.att = self
                .run_state
                .att
                .clone()
                .slice_assign([0..position + 1], scores.clone());

            let prev_value = self
                .run_state
                .value_cache
                .clone()
                .slice([
                    layer_id..layer_id + 1,
                    0..position + 1,
                    (head_index * self.head_size) as usize
                        ..((head_index + 1) * self.head_size) as usize,
                ])
                .squeeze::<2>(0)
                .transpose();
            let weighted = prev_value.matmul(scores.unsqueeze_dim(1));

            self.run_state.xb = self.run_state.xb.clone().slice_assign(
                [(head_index * self.head_size) as usize
                    ..((head_index + 1) * self.head_size) as usize],
                weighted.squeeze(1),
            )
        }
    }

    /// apply RoPE rotation to the q and k vectors for each head
    /// x is either query or key
    pub fn rope(
        x: &mut Tensor<B, 1>,
        n_heads: usize,
        head_size: usize,
        position: usize,
        device: &DefaultDevice,
    ) {
        for head_index in 0..n_heads {
            let q: Tensor<NdArray, 1> = x.clone().slice([
                (head_index * head_size) as usize..((head_index + 1) * head_size) as usize
            ]);

            for i in (0..q.dims()[0]).step_by(2) {
                // compute the RoPE embedding
                // sligtly different from loading from bin file
                let freq = 1. / 10000f32.powf(2. * (i as f32) / (head_size as f32));
                let val = position as f32 * freq;
                let fcr: f32 = val.cos();
                let fci = val.sin();

                // NOTE, this is relative index within each head
                let index = Tensor::from_ints([i as i32], device);
                let prev_val = q.clone().select(0, index.clone());
                let next_val = q.clone().select(0, index.clone() + 1);

                let updated_prev_val = prev_val.clone() * fcr - next_val.clone() * fci;
                let updated_next_val: Tensor<NdArray, 1> = prev_val * fci + next_val * fcr;

                // NOTE, need to use absolute index
                *x = x.clone().slice_assign(
                    [i + head_index * head_size..i + head_index * head_size + 1],
                    updated_prev_val,
                );
                *x = x.clone().slice_assign(
                    [i + head_index * head_size + 1..i + head_index * head_size + 2],
                    updated_next_val,
                );
            }
        }
    }

    pub fn get_next_token(&self, temperature: f32) -> usize {
        let logits = if temperature < 1e-2 {
            self.run_state.logits.clone()
        } else {
            self.run_state.logits.clone() / temperature
        };
        logits.clone().argmax(0).into_scalar().elem::<i32>() as usize
    }
}
