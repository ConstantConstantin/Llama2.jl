"""
    TokenIndex(str::AbstractVector{UInt8}, id::Integer)

Create a `TokenIndex` from a byte vector and an integer identifier.

The byte sequence is converted to `Vector{UInt8}` and the ID is
converted to `Int16`.  
Throw a `DomainError` if `id ≤ 0`.


# Examples
```jldoctest
julia> using Llama2;

julia> TokenIndex([0x61], 1)
TokenIndex(UInt8[0x61], 1)

julia> TokenIndex([0x61], -1)
ERROR: DomainError with Token index must be > 0.
[...]
```

# Developer Notes
This is an internal struct.

"""
struct TokenIndex

    str::Vector{UInt8}
    id::Int16

    function TokenIndex(str::AbstractVector{UInt8}, id::Integer)
        id < 0 && throw(DomainError("Token index must be > 0."))
        new(convert(Vector{UInt8}, str), convert(Int16, id))
    end

end

"""
    compare_tokens(first_token::TokenIndex, second_token::TokenIndex) -> Bool

Compare two `TokenIndex` objects for equality **based solely on their
byte-string contents**.  
Return `true` if both tokens contain identical `str` fields, regardless of ID.

# Examples
```jldoctest
julia> using Llama2;

julia> compare_tokens(TokenIndex([0x61], 1), TokenIndex([0x61], 2))
true

julia> compare_tokens(TokenIndex([0x61], 1), TokenIndex([0x62], 1))
false
```

"""
function compare_tokens(first_token::TokenIndex, second_token::TokenIndex)::Bool
    return (first_token.str == second_token.str)
end

"""
    Tokenizer

Construct a tokenizer storing vocabulary entries, scores, and byte-piece mappings.

# Constructors
- `Tokenizer(vocab, vocab_scores, sorted_vocab, vocab_size, max_token_length, byte_pieces)`  
  Construct a tokenizer directly from the provided fields.  
  Validate that `max_token_length > 0` and that `byte_pieces` has length 256.

- `Tokenizer(path::String, vocab_size::Integer)`  
  Load a tokenizer from a binary file.

# Fields
- `vocab`: Token byte sequences.  
- `vocab_scores`: Scores for each token.  
- `sorted_vocab`: Sorted token indices.  
- `vocab_size`: Number of vocabulary entries.  
- `max_token_length`: Maximum token length in bytes.  
- `byte_pieces`: Byte mapping (length 256).
"""
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
