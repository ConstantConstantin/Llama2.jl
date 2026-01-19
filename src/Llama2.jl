module Llama2

using StatsBase: wsample
using LinearAlgebra: dot

export Tokenizer, talktollm

include("structs.jl")
include("tokenizer.jl")
include("decode_transformer.jl")
include("forward.jl")

"""
    talktollm(modelpath::String, vocabpath::String, prompt::String, max_tokens::Int)

Generate text using a pretrained LLama2 transformer model. The function loads the
model from `modelpath` and the corresponding tokenizer from `vocabpath`. It takes an initial 
`prompt` string to start the text generation and generates up to `max_tokens` tokens.

```julia
julia> talktollm("/PATH/TO/MODEL.bin"
                ,"/PATH/TO/VOCAB.bin"
                ,"hey whats up?", 100);
```
"""
function talktollm(modelpath::String, vocabpath::String, prompt::String, max_tokens::Int=255)
    
    transformer = Transformer(modelpath)
    tok = Tokenizer(vocabpath, transformer.config.vocab_size)

    input_tokens = encode(tok, prompt)

    if isempty(input_tokens)
        input_tokens = [2] # default for empty prompt
    end

    token = input_tokens[1]
    n_input_tokens = length(input_tokens)

    for pos in 1:max_tokens

        logits = forward!(transformer, Int32(token), Int32(pos))

        if pos < n_input_tokens
            next = input_tokens[pos + 1]
        else
            softmax!(logits)
            next = wsample(logits)
        end

        next == 2 && break
        
        print(tok.vocab[next])

        token = next
        
    end

    return nothing
end

end
