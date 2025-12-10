```@meta
CurrentModule = Llama2
```

# Llama2.jl

## What is Llama2?

LLama2 is a family of pre-trained LLMs by Meta AI. More information can be found at: https://www.llama.com/

## What is Llama2.jl?

[Llama2.jl](https://github.com/ConstantConstantin/Llama2.jl) can inference a given model from within `julia`. For this cause you will have to provide your own model checkpoint. This project follows the procedure outlined by the `run.c` file from [llama2.c](https://github.com/karpathy/llama2.c).

## Getting started

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

```@index
```

```@autodocs
Modules = [Llama2]
```
