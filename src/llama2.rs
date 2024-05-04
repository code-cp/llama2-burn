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
    state_config: StateConfig,
    model_dir: Option<PathBuf>,
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

        let rope_dim = self.dim / (self.n_heads * 2);

        let freq_cis_real = Tensor::zeros(
            Shape::new([self.seq_len as usize, rope_dim as usize]),
            device,
        );

        let freq_cis_imag = Tensor::zeros(
            Shape::new([self.seq_len as usize, rope_dim as usize]),
            device,
        );

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
            freq_cis_real,
            freq_cis_imag,
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

    pub fn init_from_weights(&self, device: &DefaultDevice) -> Model {
        let mut input = BufReader::new(
            File::open(self.model_dir.clone().expect("weights path should exist"))
                .expect("weight file should exist"),
        );

        let mut read_weights = |size: (usize, usize, usize)| {
            let num = size.0 * size.1 * size.2;
            let mut bytes = Vec::new();
            for _ in 0..num {
                let byte = read_f32(&mut input).expect("weights should have correct format");
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

        let rope_dim = self.dim / (self.n_heads * 2);

        let freq_cis_real_arr =
            read_weights((self.seq_len as usize, rope_dim as usize, 1)).slice_move(s2);
        let freq_cis_real_arr = NdArrayTensor::<f32, 2>::new(freq_cis_real_arr.into_dyn().into());
        let freq_cis_real = Tensor::from_primitive(freq_cis_real_arr);

        let freq_cis_imag_arr =
            read_weights((self.seq_len as usize, rope_dim as usize, 1)).slice_move(s2);
        let freq_cis_imag_arr = NdArrayTensor::<f32, 2>::new(freq_cis_imag_arr.into_dyn().into());
        let freq_cis_imag = Tensor::from_primitive(freq_cis_imag_arr);

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
            freq_cis_real,
            freq_cis_imag,
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
    /// RoPE relatively positional embeddings
    /// (seq_len, dim/(num_heads*2))
    pub freq_cis_real: Tensor<B, 2>,
    pub freq_cis_imag: Tensor<B, 2>,
}

#[derive(Debug, Clone, Module, Shrinkwrap)]
pub struct Model {
    #[shrinkwrap(main_field)]
    pub weights: ModelWeights,
    pub run_state: RunState,
    pub n_heads: i32,
    pub head_size: i32,
    pub n_layers: i32,
    pub rms_norm: RmsNorm<B>,
}

impl Model {
    pub fn forward(&mut self, token_id: usize, position: i32) {
        let device = <NdArray as Backend>::Device::default();

        let token_embedding_index = Tensor::<B, 1, Int>::from_data(
            Data::new(vec![token_id as i32], Shape::new([1])).convert(),
            &device,
        );
        self.run_state.x = self
            .token_embedding
            .clone()
            .select(0, token_embedding_index)
            .unsqueeze();

        let position_embedding_index = Tensor::<B, 1, Int>::from_data(
            Data::new(vec![position], Shape::new([1])).convert(),
            &device,
        );
        let freq_cis_real_row: Tensor<B, 1> = self
            .freq_cis_real
            .clone()
            .select(0, position_embedding_index.clone())
            .unsqueeze();
        let freq_cis_imag_row: Tensor<B, 1> = self
            .freq_cis_imag
            .clone()
            .select(0, position_embedding_index)
            .unsqueeze();

        for l in 0..self.n_layers {
            let layer_index = Tensor::<B, 1, Int>::from_data(
                Data::new(vec![l], Shape::new([1])).convert(),
                &device,
            );
            let att_weight: Tensor<B, 1> = self
                .rms_att_weight
                .clone()
                .select(0, layer_index.clone())
                .unsqueeze();
            self.run_state.xb = self.rms_norm.forward(self.run_state.x.clone());

            // attention matmul
            self.run_state.q = self.run_state.xb.clone()
                * self.wq.clone().select(0, layer_index.clone()).unsqueeze();
            self.run_state.k = self.run_state.xb.clone()
                * self.wk.clone().select(0, layer_index.clone()).unsqueeze();
            self.run_state.v = self.run_state.xb.clone()
                * self.wv.clone().select(0, layer_index.clone()).unsqueeze();
        }
    }

    /// apply RoPE rotation to the q and k vectors for each head
    /// x is either query or key
    pub fn rope(
        x: Tensor<B, 1>,
        freq_cis_real_row: Tensor<B, 1>,
        freq_cis_imag_row: Tensor<B, 1>,
        n_heads: usize,
        head_size: usize,
        position: usize,
        device: DefaultDevice,
    ) {
        for head_index in 0..n_heads {
            let q = x.clone().slice([
                (head_index * head_size) as usize..((head_index + 1) * head_size) as usize
            ]);
            for i in (0..q.dims()[0]).step_by(2) {
                let index = Tensor::<B, 1, Int>::from_data(
                    Data::new(vec![i as i32], Shape::new([1])).convert(),
                    &device,
                );
                let prev_val = q.clone().select(0, index.clone());
                let next_val = q.clone().select(0, index.clone() + 1);

                let updated_prev_val = prev_val.clone()
                    * freq_cis_real_row.clone().select(0, index.clone())
                    - next_val.clone() * freq_cis_imag_row.clone().select(0, index.clone());
                let updated_next_val: Tensor<NdArray, 1> = prev_val
                    * freq_cis_imag_row.clone().select(0, index.clone())
                    + next_val * freq_cis_real_row.clone().select(0, index.clone());
            }
        }
    }
}
