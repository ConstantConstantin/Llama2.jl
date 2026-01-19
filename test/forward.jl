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

    # TODO: move to test of structs, when available
    @testset "Transformer Data from file" begin

        t = Llama2.Transformer(normpath(joinpath(@__DIR__, "..", "data", "stories15M.bin")))

        @testset "Transformer.Config" begin

            c = t.config

            @test c.dim == 288
            @test c.hidden_dim == 768
            @test c.n_layers == 6
            @test c.n_heads == 6
            @test c.n_kv_heads == 6
            @test c.vocab_size == 32000
            @test c.seq_len == 256

        end

        @testset "Transformer.TransformerWeights" begin

            w = t.weights

            # testing every single value is too much
            # we resort to testing single values at the beginning and along the axes of the containers
            # this assures correct reading of a value, correct location within the data and
            # correct interpretation of the axes within a container

            @test w.token_embedding_table[1, 1] == -0.059824646f0
            @test w.token_embedding_table[1, 2] == -0.014561243f0
            @test w.token_embedding_table[2, 1] == -0.017096385f0

            @test w.rms_att_weight[1, 1] == 0.79829437f0
            @test w.rms_att_weight[1, 2] == 0.7969791f0
            @test w.rms_att_weight[2, 1] == 1.1206539f0

            @test w.rms_ffn_weight[1, 1] == 1.0146376f0
            @test w.rms_ffn_weight[1, 2] == 0.9703832f0
            @test w.rms_ffn_weight[2, 1] == 1.1049336f0

            @test w.wq[1, 1, 1] == 0.03963275f0
            @test w.wq[1, 1, 2] == -0.063102335f0
            @test w.wq[1, 2, 1] == -0.0045094043f0
            @test w.wq[2, 1, 1] == 0.005027403f0

            @test w.wk[1, 1, 1] == 0.021850083f0
            @test w.wk[1, 1, 2] == 0.0113122715f0
            @test w.wk[1, 2, 1] == -5.9539005f-5
            @test w.wk[2, 1, 1] == 0.012895294f0

            @test w.wv[1, 1, 1] == -0.012940487f0
            @test w.wv[1, 1, 2] == -0.020094391f0
            @test w.wv[1, 2, 1] == 0.01877362f0
            @test w.wv[2, 1, 1] == -0.0013144497f0

            @test w.wo[1, 1, 1] == 0.009332931f0
            @test w.wo[1, 1, 2] == -0.023395995f0
            @test w.wo[1, 2, 1] == 0.011191372f0
            @test w.wo[2, 1, 1] == -0.01163089f0

            @test w.w1[1, 1, 1] == -0.03274832f0
            @test w.w1[1, 1, 2] == -0.021039095f0
            @test w.w1[1, 2, 1] == -0.030032795f0
            @test w.w1[2, 1, 1] == -0.00537954f0

            @test w.w2[1, 1, 1] == 0.0029473635f0
            @test w.w2[1, 1, 2] == 0.02511926f0
            @test w.w2[1, 2, 1] == 0.009355266f0
            @test w.w2[2, 1, 1] == 0.027657198f0

            @test w.w3[1, 1, 1] == -0.015980158f0
            @test w.w3[1, 1, 2] == 0.004169304f0
            @test w.w3[1, 2, 1] == -0.00055685936f0
            @test w.w3[2, 1, 1] == -0.015687676f0

            @test w.rms_final_weight[1] == 7.676849f0
            @test w.rms_final_weight[2] == 7.1879797f0

            @test w.wcls[1, 1] == -0.059824646f0
            @test w.wcls[1, 2] == -0.014561243f0
            @test w.wcls[2, 1] == -0.017096385f0

        end

    end

    @testset "forward!" begin

        t = Llama2.Transformer(normpath(joinpath(@__DIR__, "..", "data", "stories15M.bin")))

        l = Llama2.forward!(t, Int32(2), Int32(1))

        @test l isa Vector{Float32}
        @test l == t.state.logits

        @test length(l) == 32000
        @test l[1] == -6.727445f0
        @test l[2] == 0.8421467f0
        @test l[32000] == -6.7272153f0

        @test t.state.key_cache[1, 1, 1] == 0.009520381f0
        @test t.state.key_cache[6, 1, 1] == 0.13823175f0
        @test t.state.key_cache[1, 2, 1] == 0.0046572713f0
        @test t.state.key_cache[1, 1, 69] == -2.0702043f0

        @test t.state.value_cache[1, 1, 1] == -0.013249561f0
        @test t.state.value_cache[6, 1, 1] == -0.031174064f0
        @test t.state.value_cache[1, 2, 1] == 0.0035178794f0
        @test t.state.value_cache[1, 1, 69] == -0.09122322f0

    end

end
