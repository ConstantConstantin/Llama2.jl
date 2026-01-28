module Llama2

using StatsBase: wsample
using LinearAlgebra: dot
using Random: MersenneTwister, rand

export talktollm, chatwithllm, ChatBot

include("tokenizer.jl")
include("structs.jl")
include("decode_transformer.jl")
include("forward.jl")
include("talk.jl")

end
