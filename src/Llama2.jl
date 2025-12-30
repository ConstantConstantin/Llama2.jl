module Llama2

using StatsBase: wsample

export Tokenizer, TokenIndex, isless_tokens, str_lookup, encode, ProbIndex, Sampler, sample_mult, isless_probindex, sample_topp, talktollm
# Write your package code here.
include("structs.jl")
include("tokenizer.jl")
include("sampler.jl")
include("decode_transformer.jl")
include("forward.jl")

"""
    talktollm(modelpath::String, vocabpath::String, prompt::String, max_tokens::Int)

Generate text using a pretrained LLama2 transformer model. The function loads the
model from `modelpath` and the corresponding tokenizer from `vocabpath`. It takes an initial 
`prompt` string to start the text generation and generates up to `max_tokens` tokens.

```julia
julia> using Llama2;

julia> talktollm("/PATH/TO/MODEL.bin"
                ,"/PATH/TO/VOCAB.bin"
                ,"hey whats up?", 100);
```
"""
function talktollm(modelpath::String, vocabpath::String, prompt::String, max_tokens::Int=50)
    
    transformer = Transformer(modelpath)
    tok = Tokenizer(vocabpath, transformer.config.vocab_size-268)

    input_tokens = encode(tok, prompt)

    token = 1 # default for empty prompt

    n_input_tokens = length(input_tokens)

    if !isempty(input_tokens)
        token = input_tokens[1]
    end

    for pos in 1:max_tokens + n_input_tokens

        logits = forward!(transformer, Int32(token), Int32(pos))
        
        if pos <= n_input_tokens
            next = input_tokens[pos]
        else
            next = wsample(1:transformer.config.vocab_size, logits)
        end

        print(tok.vocab[next])

        token = next
        
    end

    return nothing
end

end
