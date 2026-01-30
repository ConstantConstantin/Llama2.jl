module Llama2

using StatsBase: wsample
using LinearAlgebra: dot

export talktollm, LlamaChat, talk!, reset!

include("structs.jl")
include("tokenizer.jl")
include("decode_transformer.jl")
include("forward.jl")
include("talk.jl")

end
