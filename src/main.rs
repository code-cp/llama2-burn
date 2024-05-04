use rand::Rng;
use std::cmp;
use std::env;
use std::fs::File;
use std::io::{self, BufReader, ErrorKind, Read, Write};
use std::path::Path;

use llama2_rs::config::*;
use llama2_rs::llama2::*;
use llama2_rs::tokenizer::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        panic!(
            "Usage: {} <checkpoint_file> [tokenizer_path] [prompt] [temperature] [max_steps]",
            &args[0]
        )
    }

    let model_path: String = String::from(&args[1]);
    if !Path::new(&model_path).exists() {
        panic!("Model path does not exist.");
    }
    let mut input =
        BufReader::new(File::open(model_path.clone()).expect("Should be able to open model file"));
    let config = StateConfig::from_binary(&mut input);

    let llama2_config = ModelConfig {
        state_config: config,
        model_dir: Some(Path::new(&model_path).to_path_buf()),
    };
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let llama2_model = llama2_config.init_from_weights(&device);

    let token_path: String = match args.get(2) {
        Some(s) => s.to_string(),
        None => String::from("tokenizer.bin"),
    };
    if !Path::new(&token_path).exists() {
        panic!("Tokenizer path does not exist.");
    }

    let tokenizer =
        Tokenizer::new(&token_path, llama2_config.vocab_size).expect("Should load tokenizer");
    let prompt: Vec<usize> = match args.get(3) {
        Some(s) => tokenizer.encode(s, true, true),
        None => tokenizer.encode("Hello, world!", true, true),
    };

    let temperature = match args.get(4) {
        Some(t) => t
            .parse::<f32>()
            .expect("Temperature should be a valid float"),
        None => 0.0,
    };

    let max_steps = match args.get(5) {
        Some(s) => s
            .parse::<i32>()
            .expect("Max steps should be a valid integer"),
        None => 20,
    };

    inference(prompt, max_steps, temperature);
}
