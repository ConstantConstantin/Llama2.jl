@testset "forward" begin

    @testset "helpers" begin

        empty_vector = Vector{Float32}(undef, 0)
        x4 = [1.0f0, 2, 3, 4]
        y4 = [1.0f0, 1, 1, 1]
        x5 = [1.0f0, 5, 4, 3, 2]
        y5 = [1.0f0, 1, 1, 1, 1]
    
        @testset "rmsnorm" begin

            @test Llama2.rmsnorm(x4, y4) isa AbstractVector{Float32}
        
            @test Llama2.rmsnorm(x4, y4) == Float32[0.36514813, 0.73029625, 1.0954444, 1.4605925]
            @test Llama2.rmsnorm(x4, x4) == Float32[0.36514813, 1.4605925, 3.2863333, 5.84237]
            @test Llama2.rmsnorm(y4, y4) == Float32[0.999995, 0.999995, 0.999995, 0.999995]
            @test Llama2.rmsnorm(x5, y5) == Float32[0.3015112, 1.507556, 1.2060448, 0.9045336, 0.6030224]
            @test Llama2.rmsnorm(x5, x5) == Float32[0.3015112, 7.53778, 4.824179, 2.7136009, 1.2060448]
            @test Llama2.rmsnorm(y5, y5) == Float32[0.999995, 0.999995, 0.999995, 0.999995, 0.999995]

            @test_throws DimensionMismatch Llama2.rmsnorm(x4, y5)
            @test_throws ArgumentError Llama2.rmsnorm(empty_vector, empty_vector)

        end

        @testset "softmax!" begin

            @test Llama2.softmax!(x4) isa AbstractVector{Float32}
            @test x4 == Float32[0.032058604, 0.08714432, 0.23688284, 0.6439143]
            Llama2.softmax!.((y4, x5, y5))
            @test y4 == Float32[0.25, 0.25, 0.25, 0.25]
            @test x5 == Float32[0.01165623, 0.63640857, 0.23412165, 0.08612854, 0.031684916]
            @test y5 == Float32[0.2, 0.2, 0.2, 0.2, 0.2]

            @test_throws ArgumentError Llama2.softmax!(empty_vector)

        end

    end

    @testset "forward!" begin

        t = Llama2.Transformer(normpath(joinpath(@__DIR__, "..", "data", "stories15M.bin")))

        l = Llama2.forward!(t, Int32(2), Int32(1))

        @test l isa Vector{Float32}
        @test l == t.state.logits
        

        #added tolerances for floating point comparisons this should be revisted maybe later
        @test length(l) == 32000
        @test isapprox(l[1], -6.727445f0; rtol=1e-2, atol=1e-5)
        @test isapprox(l[2], 0.8421467f0; rtol=1e-2, atol=1e-5)
        @test isapprox(l[32000], -6.7272153f0; rtol=1e-2, atol=1e-5)

        @test isapprox(t.state.key_cache[1, 1, 1], 0.009520381f0; rtol=1e-2, atol=1e-5)
        @test isapprox(t.state.key_cache[6, 1, 1], 0.13823175f0; rtol=1e-2, atol=1e-5)
        @test isapprox(t.state.key_cache[1, 1, 69], -2.0702043f0; rtol=1e-2, atol=1e-5)

        @test isapprox(t.state.value_cache[1, 1, 1], -0.013249561f0; rtol=1e-2, atol=1e-5)
        @test isapprox(t.state.value_cache[6, 1, 1], -0.031174064f0; rtol=1e-2, atol=1e-5)
        @test isapprox(t.state.value_cache[1, 1, 69], -0.09122322f0; rtol=1e-2, atol=1e-5)

    end

end
