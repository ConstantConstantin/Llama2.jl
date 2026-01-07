using Llama2
using Test

@testset "Llama2.jl" begin
    
    # Write your tests here. 
end

@testset "structs" begin
    #config tests
    c = Config(512, 2048, 12, 8, 8, 32000, 1024)
    @test c isa Config
    @test c.n_heads isa Int32
 
    @test_throws MethodError Config(1, 2, 3,)
    @test_throws MethodError Config(1, 2, 3,4,5,6,7,8)

    # TransformerWeights tests
    
    t = TransformerWeights(
        rand(Float32, 10, 20),
        rand(Float32, 4, 20),
        rand(Float32, 4, 20),
        rand(Float32, 4, 20, 20),
        rand(Float32, 4, 20, 20),
        rand(Float32, 4, 20, 20),
        rand(Float32, 20, 20, 4),
        rand(Float32, 4, 20, 80),
        rand(Float32, 4, 80, 20),
        rand(Float32, 4, 20, 80),
        rand(Float32, 20),
        rand(Float32, 20, 10)
    )

    @test t isa TransformerWeights
    @test t.token_embedding_table isa Matrix{Float32}
    @test t.wq isa Array{Float32,3}
    @test t.rms_final_weight isa Vector{Float32}

    @test_throws MethodError TransformerWeights(
        rand(Float32, 10, 20),
        rand(Float32, 4, 20),
        rand(Float32, 4, 20),
        rand(Float32, 4, 20, 20),
        rand(Float32, 4, 20, 20),
        rand(Float32, 4, 20, 20),
        rand(Float32, 20, 20, 4),
        rand(Float32, 4, 20, 80),
        rand(Float32, 4, 80, 20),
        rand(Float32, 4, 20, 80),
        rand(Float32, 20)
    )

    # RunState tests
    r = RunState(
        rand(Float32, 4),   # x
        rand(Float32, 4),   # xb
        rand(Float32, 4),   # xb2
        rand(Float32, 8),   # hb
        rand(Float32, 8),   # hb2
        rand(Float32, 4),   # q
        rand(Float32, 4),   # k
        rand(Float32, 4),   # v
        rand(Float32, 4, 4),# att
        rand(Float32, 10),  # logits
        rand(Float32, 2, 4, 4), # key_cache 
        rand(Float32, 2, 4, 4)  # value_cache
    )

    @test r isa RunState
    @test r.x isa Vector{Float32}
    @test r.att isa Matrix{Float32}
    @test r.key_cache isa Array{Float32,3}
end

@testset "Transformer" begin

end
@testset "tokenizer" begin

end

@testset "decode" begin
    
end
@testset "forward" begin
    
end