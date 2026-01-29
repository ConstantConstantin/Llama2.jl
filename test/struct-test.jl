

@testset "structs" begin
    # Basic tests  — nothing fancy, just type/shape sanity.

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

    @testset "Transformer" begin
        # Build a weights container with random Float32 arrays matching expected ranks.
        path = normpath(joinpath(@__DIR__, "..", "data", "stories15M.bin"))
        t = Llama2.Transformer(path)
        
        cfg = t.config
        w = t.weights
        @testset "Transformer.Config" begin
            # config sanity
            @test cfg.dim == 288
            @test cfg.hidden_dim == 768
            @test cfg.n_layers == 6
            @test cfg.n_heads == 6
            @test cfg.n_kv_heads == 6
            @test cfg.vocab_size == 32000
            @test cfg.seq_len == 256

            @test cfg.dim % cfg.n_heads == 0
        end

        @testset "Transformer.Weights-shapes" begin
            # Derived sizes used for expected weight shapes.
            head_size = div(cfg.dim, cfg.n_heads)
            nl = Int(cfg.n_layers)

            # Shape checks for the main weight tensors.
            @test size(w.token_embedding_table) == (cfg.vocab_size, cfg.dim)
            @test size(w.rms_att_weight) == (nl, cfg.dim)
            @test size(w.wq) == (nl, cfg.dim, cfg.n_heads * head_size)
            @test size(w.wk) == (nl, cfg.dim, cfg.n_kv_heads * head_size)
            @test size(w.wv) == (nl, cfg.dim, cfg.n_kv_heads * head_size)
            @test size(w.wo) == (nl, cfg.n_heads * head_size, cfg.dim)
            @test size(w.rms_ffn_weight) == (nl, cfg.dim)

            @test size(w.w1) == (nl, cfg.hidden_dim, cfg.dim)
            @test size(w.w2) == (nl, cfg.dim, cfg.hidden_dim)
            @test size(w.w3) == (nl, cfg.hidden_dim, cfg.dim)

            # Final RMS weight should be a vector over model dim.
            @test length(w.rms_final_weight) == cfg.dim
        end
        
        @testset "Transformer.Weights-values" begin

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
            @test w.wo[2, 1, 1] == -0.01163089f0
            @test w.wo[1, 2, 1] == 0.011191372f0
            
            @test w.w1[1, 1, 2] == -0.021039095f0
            @test w.w1[1, 1, 1] == -0.03274832f0
            @test w.w1[2, 1, 1] == -0.00537954f0
            @test w.w1[1, 2, 1] == -0.030032795f0

            @test w.w2[1, 1, 2] == 0.02511926f0
            @test w.w2[1, 1, 1] == 0.0029473635f0
            @test w.w2[2, 1, 1] == 0.027657198f0
            @test w.w2[1, 2, 1] == 0.009355266f0

            @test w.w3[1, 1, 2] == 0.004169304f0
            @test w.w3[1, 1, 1] == -0.015980158f0
            @test w.w3[2, 1, 1] == -0.015687676f0
            @test w.w3[1, 2, 1] == -0.00055685936f0

            @test w.rms_final_weight[2] == 7.1879797f0
            @test w.rms_final_weight[1] == 7.676849f0

            @test w.wcls[1, 2] == -0.014561243f0
            @test w.wcls[1, 1] == -0.059824646f0
            
            @test w.wcls[2, 1] == -0.017096385f0
        end
    end 

    @testset "runstate" begin
        # RunState holds temporary buffers/caches used during decoding.
        r = Llama2.RunState(
            fill(1f0, 10),
            fill(2f0, 4, 10, 20),
            fill(3f0, 4, 10, 20),
        )

        # Basic type sanity for fields.
        @test r isa Llama2.RunState
        @test r.logits isa Vector{Float32}
        @test r.key_cache isa Array{Float32,3}
        @test r.value_cache isa Array{Float32,3}
    end
end