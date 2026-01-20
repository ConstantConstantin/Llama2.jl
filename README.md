# Llama2

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ConstantConstantin.github.io/Llama2.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ConstantConstantin.github.io/Llama2.jl/dev/)
[![Build Status](https://github.com/ConstantConstantin/Llama2.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ConstantConstantin/Llama2.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ConstantConstantin/Llama2.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ConstantConstantin/Llama2.jl)

Welcome to LLama2.jl, a `julia` package that is dedicated to porting the LLama2 experience from `C`. It is  designed to generate output based on a pre-trained model.

## Installation

Start julia, activate a desired environment and add the package; it can then be loaded in your session:

```julia
pkg> add https://github.com/ConstantConstantin/Llama2.jl

julia> using Llama2
```

## Example usage

```julia
julia> print(talktollm("/PATH/TO/YOUR/MODEL.bin"))
 Once upon a time, there was a humble cat named Tom. Tom was not big or fancy, but he was very kind. He would always help others and share his toys with them.
One day, a little bird named Sally came to Tom's pond. Sally told Tom that her baby bird was sick. Tom wanted to help her, so he had an idea. He took the bird home and gave her food. The bird seemed happy, but one day, they had to leave a big mess in the pond.
Sally found a magic wand. She waved it, and suddenly, the baby bird vanished! Tom was very sad and went back to the pond. But now, he was ready to help. He waved his magic wand, and the baby bird came back! Tom was so happy to see Sally and helped her with her baby bird. From that day on, they all became good friends and lived happily ever after.
```

Please refer to the docs for more detailed information on how to use this package.

## Links

### Meta Datasets for diffrent sizes
https://huggingface.co/meta-llama

### Original Repo of Llama2 in C
https://github.com/karpathy/llama2.c


