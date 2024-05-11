use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use llama2_rs::config::*;
use llama2_rs::llama2::*;
use llama2_rs::tokenizer::*;

fn main() {
    let model_path: String = String::from("./data/stories15M.bin");
    // let model_path: String = String::from("./data/stories110M.bin");
    if !Path::new(&model_path).exists() {
        panic!("Model path does not exist.");
    }
    let mut input =
        BufReader::new(File::open(model_path.clone()).expect("Should be able to open model file"));
    let config = StateConfig::from_binary(&mut input);

    let llama2_config = ModelConfig {
        state_config: config,
    };
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let mut llama2_model = llama2_config.init_from_weights(&device, &mut input);

    let token_path: String = String::from("./data/tokenizer.bin");
    if !Path::new(&token_path).exists() {
        panic!("Tokenizer path does not exist.");
    }

    let tokenizer =
        Tokenizer::new(&token_path, llama2_config.vocab_size).expect("Should load tokenizer");
    let mut string_seq = "One day, Lily met a Shoggoth".to_string();
    let input_len = string_seq.len();
    // Initialize with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    let prompt = tokenizer.encode(string_seq.as_str(), true, false);
    // println!("Prompt: {:?}", prompt);

    // let temperature = 0.;
    let temperature = 0.5;

    let now = Instant::now();
    let mut prev_token = 0;
    for (position, token) in prompt.iter().enumerate() {
        llama2_model.forward(*token, position as i32);
        prev_token = *token;
    }
    for i in 0..llama2_config.seq_len - prompt.len() as i32 {
        let next_token = llama2_model.get_next_token(temperature);
        string_seq.push_str(tokenizer.decode(prev_token, next_token).as_str());
        prev_token = next_token;
        llama2_model.forward(next_token as usize, prompt.len() as i32 + i);
    }

    println!(
        "\nRan at {} tok/s.",
        ((string_seq.len() - input_len) as f32) / now.elapsed().as_secs_f32()
    );
    println!("result: {string_seq}");
}
