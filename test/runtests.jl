using Llama2
using Test

@testset "Llama2.jl" begin
    include("forward-test.jl")
    include("talk-test.jl")
    include("struct-test.jl")
    include("tokenizer-test.jl")
    include("transformer-test.jl")
end