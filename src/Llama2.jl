module Llama2

include("tokenizer.jl")

export Tokenizer, TokenIndex, compare_tokens
# Write your package code here.
include("structs.jl")
include("tokenizer.jl")

export Tokenizer, TokenIndex, compare_tokens, str_lookup, encode
# Write your package code here.
include("structs.jl")

end
