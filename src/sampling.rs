use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

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

use crate::config::DefalutBackend as B;

pub fn get_next_token(logits: &Tensor<B, 1>, temperature: f32) -> usize {
    let logits = if temperature < 1e-2 {
        logits.clone()
    } else {
        logits.clone() / temperature
    };

    if temperature >= 0.99 {
        logits.clone().argmax(0).into_scalar().elem::<i32>() as usize
    } else {
        let prob = activation::softmax(logits, 0);
        // println!("prob {:?}", prob.to_data());

        let mut probabilities: Vec<f32> = Vec::new();
        for val in prob.iter_dim(0) {
            probabilities.push(val.into_scalar().elem::<f32>());
        }
        // println!("probabilities {probabilities:?}");
        let weighted_index = WeightedIndex::new(&probabilities).unwrap();

        let mut rng = rand::thread_rng();
        let class_index = weighted_index.sample(&mut rng);

        class_index
    }
}
