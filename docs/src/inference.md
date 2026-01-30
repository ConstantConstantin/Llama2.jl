# Inference

## Prequisites

A model checkpoint is required. You can use the one provided in `data` or get other Llama2 models online:

```
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
```

## Inferencing

You can either generate a single text, optionally giving an input prompt, or have an interactive chat. For an interactive Chat the maximum tokens that can be generated is limited by the sequence length of the Model choosen, for ours its 256. 

As default there is no temperature set which will then lead to the usage of `wsample`. If a temperature is given `sample` will be used.
For more information read the the `sample.jl`documentation 

```@docs
talktollm
ChatBot
chatwithllm
```
