## initial results 

stories15M

```
Ran at 2.8592846 tok/s.
result: One day, Lily met a Shoggothotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsotsball tussenstandstandols seats seats seats bad bad bad bad
```

stories110M

```
Ran at 7.408795 tok/s.
result: One day, Lily met a Shoggothiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresiresersersersersersersersersersersersersersersersersersersersersersersersersersersersersersersersersersersersersersersers su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su su
```

## issues with tokenizer

- initially, I used hashmap for token to id and id to token mapping, then I found out the mapping from token to id is not one to one, eg 
```
Found special token 'D' at index 71
Found special token 'D' at index 29928
```

## issues with ROPE 

- I switched the cos and sin in code below
```rust
                let freq = 1. / 10000f32.powf(2. * (i as f32) / (head_size as f32));
                let val = position as f32 * freq;
                let fcr = val.cos();
                let fci = val.sin();
```

## issues with kv cache 

- I used key_cache for both key and value, which is the main bug 