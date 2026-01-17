using Llama2
using Test

@testset "Llama2.jl" begin
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

    # Tests for functions/types created in decode_transformer.jl
    @testset "Transformer" begin
        # Load model
        path = normpath(joinpath(@__DIR__, "..", "data", "stories15M.bin"))
        t = Llama2.Transformer(path)

        cfg = t.config
        w = t.weights

        # Minimal config sanity: positive sizes and valid head partition.
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

    @testset "tokenizer" begin
        @testset "TokenIndex" begin
            # TokenIndex stores a token string + its integer id.
            ti = TokenIndex("Julia", 1)
            @test ti.str == "Julia"
            @test ti.id == Int16(1)

            # ids are normalized to Int16 internally.
            ti2 = TokenIndex("X", Int64(2))
            @test ti2.id == Int16(2)

            # Negative ids are invalid.
            @test_throws DomainError TokenIndex("Bad", -1)

            # Zero is allowed 
            ti0 = TokenIndex("Zero", 0)
            @test ti0.id == Int16(0)
        end

        @testset "compare_tokens" begin
            # Comparison helper for sorting by string (and/or tie-breaking rules).
            @test compare_tokens(TokenIndex("A", 1), TokenIndex("B", 2)) == true
            @test compare_tokens(TokenIndex("B", 1), TokenIndex("A", 2)) == false

            # Same string should not be considered "less".
            @test compare_tokens(TokenIndex("aa", 1), TokenIndex("aa", 2)) == false
        end

        @testset "Tokenizer" begin
            # Build a tiny tokenizer (vocab/scores + byte pieces table).
            vocab = ["a", "b", "ab"]
            scores = Float32[0.0, 0.0, 10.0]

            # 'sorted' can start empty; encode() typically fills/sorts it.
            sorted = TokenIndex[]
            byte_pieces = collect(UInt8.(0:255))

            tok = Tokenizer(vocab, scores, sorted, 3, 10, byte_pieces)

            # Basic field sanity.
            @test tok.vocab_size == Int16(3)
            @test tok.max_token_length == UInt16(10)
            @test length(tok.byte_pieces) == 256

            # Wrong byte_pieces length should error.
            @test_throws ArgumentError Tokenizer(vocab, scores, sorted, 3, 10, UInt8[1,2,3])

            # Invalid max token length should error.
            @test_throws DomainError Tokenizer(vocab, scores, sorted, 3, -1, byte_pieces)
        end

        @testset "str_lookup" begin
            # Binary search / lookup in a sorted vocab list.
            sorted_vocab = [TokenIndex("aa", 1), TokenIndex("bb", 2), TokenIndex("cc", 3)]
            @test str_lookup("aa", sorted_vocab) == Int16(1)
            @test str_lookup("bb", sorted_vocab) == Int16(2)

            # Not found returns -1 (sentinel).
            @test str_lookup("ba", sorted_vocab) == Int16(-1)
        end

        @testset "encode" begin
            # Encode a string into token ids using the tiny vocab.
            vocab = ["a", "b", "ab"]
            scores = Float32[0.0, 0.0, 10.0]
            byte_pieces = collect(UInt8.(0:255))

            # Preallocate sorted vocab storage.
            sorted = Vector{TokenIndex}(undef, 3)
            tok = Tokenizer(vocab, scores, sorted, 3, 10, byte_pieces)

            ids = encode(tok, "ab")
            @test ids == [Int16(3)]

            # If the tokenizer cannot represent a string, encode should throw.
            sorted2 = Vector{TokenIndex}(undef, 1)
            tok2 = Tokenizer(["a"], Float32[0.0], sorted2, 1, 10, byte_pieces)
            @test_throws ArgumentError encode(tok2, "b")
        end
    end 
end
