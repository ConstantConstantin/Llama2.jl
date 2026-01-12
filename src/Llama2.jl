module Llama2

export Tokenizer, TokenIndex, compare_tokens, str_lookup, encode
# Write your package code here.

include("structs.jl")
include("tokenizer.jl")
include("decode_transformer.jl")
include("forward.jl")

function talktollm(modelpath::String, vocabpath::String, prompt::String, max_tokens::Int=50)
    
    t = Transformer(modelpath)
    tok = Tokenizer(vocabpath, t.config.vocab_size-268)
    input_tokens = encode(tok, prompt)
    
    print(input_tokens)

    forward!(t, Int32(input_tokens[1]), Int32(0))

    return nothing
end

export talktollm

end
