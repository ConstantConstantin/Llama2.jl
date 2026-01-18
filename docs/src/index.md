```@meta
CurrentModule = Llama2
```

# Llama2.jl

## What is Llama2?

LLama2 is a family of pre-trained LLMs by Meta AI. More information can be found at: https://www.llama.com/

## What is Llama2.jl?

[Llama2.jl](https://github.com/ConstantConstantin/Llama2.jl) can inference a given model from within `julia`. For this cause you will have to provide your own model checkpoint. This project follows the procedure outlined by the `run.c` file from [llama2.c](https://github.com/karpathy/llama2.c).

## Getting started

Start julia, activate a desired environment and add the package; it can then be loaded in your session:

```julia
(@v1.11) pkg> activate .

(myLlama2) pkg> add https://github.com/ConstantConstantin/Llama2.jl

julia> using Llama2
```

```@index
```
