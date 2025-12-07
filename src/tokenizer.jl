struct TokenIndex

    str::Vector{UInt8}
    id::Int16

    function TokenIndex(str::AbstractVector{UInt8}, id::Integer)
        id < 0 && throw(DomainError("Token index must be > 0."))
        new(convert(Vector{UInt8}, str), convert(Int16, id))
    end

end

function compare_tokens(first_token::TokenIndex, second_token::TokenIndex)::Bool
    return (first_token.str == second_token.str)
end

struct Tokenizer

    vocab::Vector{Vector{UInt8}}
    vocab_scores::Vector{Float32}
    sorted_vocab::Vector{TokenIndex}
    vocab_size::Int16
    max_token_length::UInt16
    byte_pieces::Vector{UInt8}

    function Tokenizer(vocab::AbstractVector{<:AbstractVector{UInt8}},
        vocab_scores::AbstractVector{Float32},
        sorted_vocab::AbstractVector{TokenIndex},
        vocab_size::Integer,
        max_token_length::Integer,
        byte_pieces::AbstractVector{UInt8}
    )
        max_token_length < 0 && throw(DomainError("max_token_length must be > 0."))
        length(byte_pieces) != 256 && throw(ArgumentError("Length of byte_pieces must be 256."))
        
        new(convert(Vector{Vector{UInt8}}, vocab), 
            convert(Vector{Float32}, vocab_scores), 
            convert(Vector{TokenIndex}, sorted_vocab), 
            convert(Int16, vocab_size), 
            convert(UInt16, max_token_length), 
            convert(Vector{UInt8}, byte_pieces)
        )
    end
end
function Tokenizer(tokenizer_path::String, vocab_size::Integer)
    byte_pieces = collect(UInt8.(0:255))
    sorted_vocab = Vector{TokenIndex}()

    vocab_scores = Vector{Float32}(undef, vocab_size)
    vocab = Vector{Vector{UInt8}}(undef, vocab_size)
    max_token_length = 0

    open(tokenizer_path) do f
        max_token_length = Int(read(f, Int32))

        for i in 1:vocab_size
            vocab_scores[i] = read(f, Float32)
            vocab_len = read(f, Int32)
            vocab[i] = read(f, vocab_len)
        end
    end

    Tokenizer(vocab, vocab_scores, sorted_vocab, vocab_size, max_token_length, byte_pieces)
end
