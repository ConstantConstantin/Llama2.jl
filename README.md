# Llama2

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ConstantConstantin.github.io/Llama2.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ConstantConstantin.github.io/Llama2.jl/dev/)
[![Build Status](https://github.com/ConstantConstantin/Llama2.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ConstantConstantin/Llama2.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ConstantConstantin/Llama2.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ConstantConstantin/Llama2.jl)

Welcome to LLama2.jl, a `julia` package that is dedicated to porting the LLama2 experience from `C`. It is  designed to generate output based on a pre-trained model.

## Installation

Clone the repository to a desired location:

```
cd /PATH/TO/DESIRED/LOCATION/
git clone git@github.com:ConstantConstantin/Llama2.git
```

Start julia, activate a desired environment and add the package; it can then be loaded in your session:

```julia
(@v1.11) pkg> activate .
  Activating new project at `PATH/TO/MY/ENVIRONMENT/myLlama2`

(myLlama2) pkg> add /PATH/TO/DESIRED/LOCATION/Llama2/
     Cloning git-repo `/PATH/TO/DESIRED/LOCATION/Llama2`
    Updating git-repo `/PATH/TO/DESIRED/LOCATION/Llama2`
    Updating registry at `~/.julia/registries/General.toml`
   Resolving package versions...
    Updating `PATH/TO/MY/ENVIRONMENT/Project.toml`
  [0e620e9f] + Llama2 v1.0.0-DEV `/PATH/TO/DESIRED/LOCATION/Llama2#aj/docs`
    Updating `PATH/TO/MY/ENVIRONMENT/Manifest.toml`
  [0e620e9f] + Llama2 v1.0.0-DEV `/PATH/TO/DESIRED/LOCATION/Llama2#aj/docs`
Precompiling project...
  1 dependency successfully precompiled in 1 seconds

julia> using Llama2
```

## Example usage

```julia
julia> "our example"
"our example"
```

Please refer to the docs for more detailed information on how to use this package.