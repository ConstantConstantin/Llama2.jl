# Developer's Corner

You want to understand how this package works or modify the code you are running? Here the necessary tools are provided and explained.

## Tokenizer

```@docs
Llama2.Tokenizer
Llama2.TokenIndex
Base.isless(::Llama2.TokenIndex, ::Llama2.TokenIndex)
Llama2.str_lookup
Llama2.encode
```

## Transformer

```@docs
Llama2.Transformer
Llama2.Config
Llama2.TransformerWeights
Llama2.RunState
```

## forward

```@docs
Llama2.forward
Llama2.rmsnorm
Llama2.softmax
```

## Sampler

```@docs
Llama2.Sampler
Llama2.ProbIndex
isless(::Llama2.ProbIndex, ::Llama2.ProbIndex)
Llama2.sample_mult
Llama2.sample_topp
```