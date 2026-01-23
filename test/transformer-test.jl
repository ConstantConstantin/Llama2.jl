using Llama2
using Test

@testset "Transformer" begin
        # Load model
        path = normpath(joinpath(@__DIR__, "..", "data", "stories15M.bin"))
        t = Llama2.Transformer(path)

        cfg = t.config
        w = t.weights

        # Minimal config sanity
        @test cfg.dim > 0
        @test cfg.n_heads > 0
        @test cfg.dim % cfg.n_heads == 0

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