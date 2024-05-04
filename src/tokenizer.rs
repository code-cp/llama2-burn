use anyhow::Result;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use crate::utils::*;

pub enum SpecialToken {
    Unk = 0,
    Bos = 1,
    Eos = 2,
}

pub struct Tokenizer {
    pub vocab_size: i32,
    pub max_token_length: i32,
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: HashMap<usize, String>,
    pub id_to_score: HashMap<usize, f32>,
    // stores all single-byte strings
    pub byte_pieces: Vec<char>,
}

impl Tokenizer {
    /// build_tokenizer in llama2.c
    pub fn new<P: AsRef<Path>>(path: P, vocab_size: i32) -> Result<Self> {
        let f = File::open(path.as_ref()).expect("tokenizer file should exist");
        let mut input = BufReader::new(f);

        let max_token_length = read_i32(&mut input)?;
        let mut token_to_id = HashMap::with_capacity(vocab_size as usize);
        let mut id_to_score = HashMap::with_capacity(vocab_size as usize);

        for i in 0..vocab_size {
            let score = read_f32(&mut input)?;
            id_to_score.insert(i as usize, score);
            let str_len = read_i32(&mut input)?;

            let mut word: String = String::new();
            for _ in 0..str_len {
                let string = read_string(&mut input)?;
                word.push_str(&string);
            }

            token_to_id.insert(word, i as usize);
        }

        let id_to_token = token_to_id
            .iter()
            .map(|(k, v)| (v.clone(), k.clone()))
            .collect();

        let byte_pieces: Vec<char> = (0..=256).map(|i| i as u8 as char).collect();

        Ok(Self {
            vocab_size,
            max_token_length,
            token_to_id,
            id_to_token,
            id_to_score,
            byte_pieces,
        })
    }

    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        let mut tokens: Vec<usize> = Vec::new();

        // if beginning, add bos token
        if bos {
            tokens.push(SpecialToken::Bos as usize);
        }

        // add dummy prefix if input is empty
        if !text.is_empty() {
            let dummy_prefix = self.token_to_id.get(" ").expect("Token should be legit");
            tokens.push(*dummy_prefix);
        }

        for ch in text.chars() {
            let string = ch.to_string();
            match self.token_to_id.get(&string) {
                Some(token_id) => tokens.push(*token_id),
                None => {
                    // byte_fallback encoding: just encode each byte as a token
                    // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                    // so the individual bytes only start at index 3
                    for byte in string.as_bytes() {
                        tokens.push(*byte as usize + 3);
                    }
                }
            }
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        let mut buffer = String::with_capacity(self.max_token_length as usize * 2);
        loop {
            let mut best_score = f32::MIN;
            let mut best_vocab_id = None;
            let mut best_token_id = None;

            for i in 0..(tokens.len() - 1) {
                buffer.clear();
                buffer.push_str(
                    &self
                        .id_to_token
                        .get(&(i as usize))
                        .expect("token id should be legit"),
                );
                buffer.push_str(
                    &self
                        .id_to_token
                        .get(&((i + 1) as usize))
                        .expect("token id should be legit"),
                );
                if let Some(&matched_id) = self.token_to_id.get(&buffer) {
                    let score = *self
                        .id_to_score
                        .get(&matched_id)
                        .expect("Score should exist");
                    if best_score < score {
                        best_score = score;
                        best_token_id = Some(matched_id);
                        best_vocab_id = Some(i);
                    }
                }
            }

            if let Some(best_id) = best_vocab_id {
                tokens[best_id] = best_token_id.unwrap();
                // merged two token become one token
                tokens.remove(best_id + 1);
            } else {
                // cannot find more merges
                break;
            }
        }

        // if end of file, add extra eos token
        if eos {
            tokens.push(SpecialToken::Eos as usize);
        }

        tokens
    }

    pub fn decode(&self, prev_token_id: usize, cur_token_id: usize) -> String {
        let mut piece = self
            .id_to_token
            .get(&cur_token_id)
            .expect("cur_token_id should exist in hashmap")
            .as_str();
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        if prev_token_id == SpecialToken::Bos as usize {
            piece = piece.trim_start();
        }
        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        // parse this and convert and return the actual byte
        if let Some(hex) = piece.strip_prefix("<0x") {
            // from_str_radix Converts a string slice in a given base to an integer.
            if let Ok(byte) = usize::from_str_radix(&hex[..2], 16) {
                return self.byte_pieces[byte].to_string();
            }
        }
        piece.to_string()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn decode_token() {
        let tokenizer = Tokenizer::new("tokenizer.bin", 32000).unwrap();

        assert_eq!(
            tokenizer.decode(SpecialToken::Unk as usize, SpecialToken::Unk as usize),
            "<unk>"
        );

        assert_eq!(
            tokenizer.decode(SpecialToken::Unk as usize, SpecialToken::Bos as usize),
            "\n<s>\n"
        );

        assert_eq!(
            tokenizer.decode(SpecialToken::Unk as usize, SpecialToken::Eos as usize),
            "\n</s>\n"
        );

        // this is space
        // println!("Expected: {:?}", tokenizer.id_to_token.get(&29871));

        // let text = "this is a dog";
        let text = &(0 as u8 as char).to_string();

        let encoded = tokenizer.encode(text, true, true);
        // println!("Encoded tokens: {:?}", encoded);
        let mut decoded = String::new();
        decoded.push_str(&tokenizer.decode(SpecialToken::Unk as usize, SpecialToken::Bos as usize));
        for (prev, cur) in encoded.iter().zip(encoded.iter().skip(1)) {
            let dec = tokenizer.decode(*prev, *cur);
            decoded.push_str(&dec);
        }
        // println!("Decoded {:?}", decoded);
        assert_eq!(decoded, format!("\n<s>\n{text}\n</s>\n"));
    }
}
