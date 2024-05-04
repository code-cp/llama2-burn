use anyhow::Result;
use std::io::{self, BufReader, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

pub fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    let mut i_buffer = [0u8; std::mem::size_of::<i32>()];
    r.read_exact(&mut i_buffer)?;
    Ok(i32::from_le_bytes(i_buffer))
}

pub fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut f_buffer = [0u8; std::mem::size_of::<f32>()];
    r.read_exact(&mut f_buffer)?;
    Ok(f32::from_le_bytes(f_buffer))
}

pub fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let mut s_buffer = [0u8; 1];
    r.read_exact(&mut s_buffer)?;
    let string = unsafe { String::from_utf8_unchecked(s_buffer.to_vec()) };
    Ok(string)
}
