## llama2 

- https://www.signalpop.com/2024/02/10/understanding-llama2-c-and-chatgpt-a-visual-design-walkthrough/
- https://www.signalpop.com/2024/02/17/understanding-llama2-c-training-a-visual-design-walkthrough/
- https://github.com/RahulSChand/llama2.c-for-dummies
- https://www.youtube.com/watch?v=Mn_9W1nCFLo&t=1931s&ab_channel=UmarJamil

## sentencepiece 

- https://towardsdatascience.com/sentencepiece-tokenizer-demystified-d0a3aac19b15
- https://medium.com/codex/sentencepiece-a-simple-and-language-independent-subword-tokenizer-and-detokenizer-for-neural-text-ffda431e704e
- https://tiktokenizer.vercel.app/?model=meta-llama%2FMeta-Llama-3-70B

## convert ndarray to tensor 

```
let tensor = Tensor::from_floats(some_ndarray.as_slice().unwrap(), device).reshape([some_ndarray.dim()]);
```

```
let arr = ArcArray::zeros((5, 6));
let tensor = NdArrayTensor::<f32, 2>::new(arr.into_dyn());
let tensor = Tensor::<NdArray, 2>::new(tensor);
```

- for training can call Tensor::from_inner to wrap the NdArrayTensor inside autodiff

## select_assign in burn 

- select_assign is += 
```
/// Assign the selected elements along the given dimension corresponding to the given indices
    /// from the value tensor to the original tensor using sum reduction.
    ///
    /// Example using a 3D tensor:
    ///
    /// `input[indices[i], j, k] += values[i, j, k]; // dim = 0`
    /// `input[i, indices[j], k] += values[i, j, k]; // dim = 1`
    /// `input[i, j, indices[k]] += values[i, j, k]; // dim = 2`
```
- slice_assign is = 