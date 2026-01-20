module Llama2

using StatsBase: wsample
using LinearAlgebra: dot

export Tokenizer, talktollm

include("structs.jl")
include("tokenizer.jl")
include("decode_transformer.jl")
include("forward.jl")

_vocabpath = normpath(joinpath(@__DIR__, "..", "data", "tokenizer.bin"))

"""
    talktollm(modelpath::String, prompt::String[, max_tokens::Int]; vocabpath::String)

Generate text using a pretrained LLama2 transformer model.
The function loads the
model from `modelpath` and the corresponding tokenizer from `vocabpath`
(which defaults to `"data/tokenizer.bin"`). It takes an initial `prompt` string
to start the text generation and generates up to `max_tokens` tokens.

```julia
julia> talktollm("/PATH/TO/YOUR/MODEL.bin")
 Once upon a time, there was a little girl named Lily. She loved to play in the park with her friends. One day, while they were playing, they saw a giant walking by. They were very fearful and didn't know what to do.<0x0A>Suddenly, a little bird flew down and landed on the giant's back. The bird started to peck at the giant, like it was his babies. Lily and her friends were scared, but the bird sang a beautiful song and the giant smiled. He was walking away with the bird on his back!<0x0A>Lily and her friends were very happy and thanked the giant for showing them the way. From that day on, they would always look up at the giant whenever they played in the park, with his fluffy tail waving in the wind once again.

julia> talktollm("/PATH/TO/YOUR/MODEL.bin", "\"What is this?\"")
"What is this?" she asked. <0x0A>Sissy smiled and said, "That's an aeroplane! We can hop on it!" <0x0A>Lucy was so excited, but she was also a little scared that the aeroplane might not be cool sooner. She laughed, but kept it a little longer. <0x0A>Sissy and Lucy both climbed into the aeroplane. Suddenly they felt like they were flying up! <0x0A>"Weird", said Lucy, smiling.<0x0A>They hopped off the aeroplane, and the grass was so soft and cool. But they were still too small to get on.<0x0A>Lucy grinned; she had so much fun exploring the world from high up in the sky!
```
"""
function talktollm(modelpath::String, prompt::String = "", max_tokens::Int=255; vocabpath::String = _vocabpath)

    transformer = Transformer(modelpath)
    tok = Tokenizer(vocabpath, transformer.config.vocab_size)

    input_tokens = encode(tok, prompt)

    if isempty(input_tokens)
        input_tokens = [2] # default for empty prompt
    else
        print(tok.vocab[input_tokens[1]])
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

    println()
    
    return nothing
end

end
