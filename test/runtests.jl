using Llama2
using Test

@testset "Llama2.jl" begin
    include("forward.jl")
    include("talk.jl")
end
