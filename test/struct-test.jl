using Llama2
using Test

@testset "structs" begin
        # Basic smoke tests for the core structs — nothing fancy, just type/shape sanity.

        @testset "Config" begin
            # Construct a valid Config and check basic typing / conversions.
            c = Llama2.Config(512, 2048, 12, 8, 8, 32000, 1024)
            @test c isa Llama2.Config

            # Fields are stored as specific integer types (here: Int32).
            @test c.n_heads isa Int32

            # Wrong arity should throw (too few / too many args).
            @test_throws MethodError Llama2.Config(1, 2, 3,)
            @test_throws MethodError Llama2.Config(1, 2, 3, 4, 5, 6, 7, 8)
        end

        @testset "TransformerWeights" begin
            # Build a weights container with random Float32 arrays matching expected ranks.
            t = Llama2.TransformerWeights(
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

            # Type checks: the struct and a few representative fields.
            @test t isa Llama2.TransformerWeights
            @test t.token_embedding_table isa Matrix{Float32}
            @test t.wq isa Array{Float32,3}
            @test t.rms_final_weight isa Vector{Float32}

            # Missing an argument should throw (constructor arity check).
            @test_throws MethodError Llama2.TransformerWeights(
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
        end

        @testset "runstate" begin
            # RunState holds temporary buffers/caches used during decoding.
            r = Llama2.RunState(
                rand(Float32, 10),
                rand(Float32, 4, 10, 20),
                rand(Float32, 4, 10, 20)
            )

            # Basic type sanity for fields.
            @test r isa Llama2.RunState
            @test r.logits isa Vector{Float32}
            @test r.key_cache isa Array{Float32,3}
            @test r.value_cache isa Array{Float32,3}
        end
    end