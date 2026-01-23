
@testset "tokenizer" begin
    @testset "TokenIndex" begin
            # TokenIndex stores a token string + its integer id.
            ti = Llama2.TokenIndex("Julia", 1)
            @test ti.str == "Julia"
            @test ti.id == Int16(1)

            # ids are normalized to Int16 internally.
            ti2 = Llama2.TokenIndex("X", Int64(2))
            @test ti2.id == Int16(2)

            # Negative ids are invalid.
            @test_throws DomainError Llama2.TokenIndex("Bad", -1)
            # Zero is allowed 
            ti0 = Llama2.TokenIndex("Zero", 0)
            @test ti0.id == Int16(0)
        end

    @testset "compare_tokens" begin
            # Comparison helper for sorting by string (and/or tie-breaking rules).
            @test Llama2.compare_tokens(Llama2.TokenIndex("A", 1), Llama2.TokenIndex("B", 2)) == true
            @test Llama2.compare_tokens(Llama2.TokenIndex("B", 1), Llama2.TokenIndex("A", 2)) == false

            # Same string should not be considered "less".
            @test Llama2.compare_tokens(Llama2.TokenIndex("aa", 1), Llama2.TokenIndex("aa", 2)) == false
        end

    @testset "Tokenizer" begin
            # IMPORTANT: vocab must be >= 14 because Tokenizer constructor does vocab[14] = "\n"
            # subject to change tho in near future but for now this is the way :)
            vocab = fill("x", 14)
            vocab[1] = "a"
            vocab[2] = "b"
            vocab[3] = "ab"

            scores = zeros(Float32, 14)
            scores[3] = 10.0f0
            # 'sorted' can start empty; encode() typically fills/sorts it.
            sorted = Llama2.TokenIndex[]
            byte_pieces = collect(UInt8.(0:255))

            tok = Llama2.Tokenizer(vocab, scores, sorted, 14, 10, byte_pieces)

            # Basic field sanity.
            @test tok.vocab_size == Int16(14)
            @test tok.max_token_length == UInt16(10)
            @test length(tok.byte_pieces) == 256

            # Wrong byte_pieces length should error.
            @test_throws ArgumentError Llama2.Tokenizer(vocab, scores, sorted, 14, 10, UInt8[1,2,3])

            # Invalid max token length should error.
            @test_throws DomainError Llama2.Tokenizer(vocab, scores, sorted, 14, -1, byte_pieces)
        end

    @testset "str_lookup" begin
            # Binary search / lookup in a sorted vocab list.
            sorted_vocab = [Llama2.TokenIndex("aa", 1), Llama2.TokenIndex("bb", 2), Llama2.TokenIndex("cc", 3)]
            @test Llama2.str_lookup("aa", sorted_vocab) == Int16(1)
            @test Llama2.str_lookup("bb", sorted_vocab) == Int16(2)

            # Not found returns -1 (sentinel).
            @test Llama2.str_lookup("ba", sorted_vocab) == Int16(-1)
        end

    @testset "encode" begin
            # Encode a string into token ids using the tiny vocab.
            vocab = fill("x", 14)
            vocab[1] = "a"
            vocab[2] = "b"
            vocab[3] = "ab"

            scores = zeros(Float32, 14)
            scores[3] = 10.0f0
            byte_pieces = collect(UInt8.(0:255))

            # Preallocate sorted vocab storage.
            sorted = Vector{Llama2.TokenIndex}(undef, 14)
            tok = Llama2.Tokenizer(vocab, scores, sorted, 14, 10, byte_pieces)

            ids = Llama2.encode(tok, "ab")
            @test ids == [Int16(3)]

            # If the tokenizer cannot represent a string, encode should throw.
            vocab2 = fill("x", 14)
            vocab2[1] = "a"

            scores2 = zeros(Float32, 14)

            sorted2 = Vector{Llama2.TokenIndex}(undef, 14)
            tok2 = Llama2.Tokenizer(vocab2, scores2, sorted2, 14, 10, byte_pieces)
            @test_throws ArgumentError Llama2.encode(tok2, "b")
    end
end 