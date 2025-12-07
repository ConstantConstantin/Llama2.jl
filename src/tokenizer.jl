"""
    struct TokenIndex

A lightweight container representing a token and its integer identifier.

# Fields
- `str::Vector{UInt8}`: The byte representation of the token.
- `id::Int16`: The numeric identifier of the token. Must be positive.

# Constructors
- `TokenIndex(str::AbstractVector{UInt8}, id::Integer)`:  
  Creates a `TokenIndex` after converting `str` to a `Vector{UInt8}` and  
  `id` to `Int16`. Throws a `DomainError` if `id ≤ 0`.

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

Compares two `TokenIndex` objects for equality **based solely on their
byte-string contents**.  
Returns `true` if both tokens contain identical `str` fields, regardless of ID.

# Examples
```julia
compare_tokens(TokenIndex([0x61], 1), TokenIndex([0x61], 2))  # true
compare_tokens(TokenIndex([0x61], 1), TokenIndex([0x62], 1))  # false
```

"""
function compare_tokens(first_token::TokenIndex, second_token::TokenIndex)::Bool
    return (first_token.str == second_token.str)
end

"""
struct Tokenizer

A tokenizer structure containing vocabulary, scores, and token metadata used for
byte-level or subword tokenization.

Fields

vocab::Vector{Vector{UInt8}}: List of token byte sequences.

vocab_scores::Vector{Float32}: Scores associated with each token (e.g., merge scores).

sorted_vocab::Vector{TokenIndex}: A sorted list of token indices (may be
populated externally after construction).

vocab_size::Int16: Number of tokens in the vocabulary.

max_token_length::UInt16: Maximum number of bytes in any token.

byte_pieces::Vector{UInt8}: A 256-element list of byte values (0-255), used
for constructing base byte tokens.

Constructors

Tokenizer(vocab, vocab_scores, sorted_vocab, vocab_size, max_token_length, byte_pieces)
Normalizes input types and validates constraints:

max_token_length must be positive.

length(byte_pieces) must equal 256.

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

"""
Tokenizer(tokenizer_path::String, vocab_size::Integer)

Loads a tokenizer from a binary file on disk.

File Format

The file is expected to contain, in order:

Int32 — maximum token length

For each token (repeated vocab_size times):

Float32 — the token's score

Int32 — length of the token in bytes

raw bytes representing the token

Returns

A fully constructed Tokenizer with:

vocab, vocab_scores, and max_token_length loaded from the file

sorted_vocab initialized empty

byte_pieces set to all 256 possible bytes

Throws

ArgumentError, EOFError, or I/O errors if the file lacks the expected structure.

"""
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
