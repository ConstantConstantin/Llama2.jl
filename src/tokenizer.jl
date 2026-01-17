"""
    TokenIndex(str::String, id::Integer)

Create a `TokenIndex` from a string and an integer identifier.

The byte sequence is converted to `String` and the ID is
converted to `Int16`.  
Throw a `DomainError` if `id ≤ 0`.


# Examples
```jldoctest
julia> using Llama2;

julia> TokenIndex("Julia", 1)
TokenIndex("Julia", 1)

julia> TokenIndex("Julia", -1)
ERROR: DomainError with Token index must be > 0.
[...]
```

# Developer Notes
This is an internal struct.

"""
struct TokenIndex

    str::String
    id::Int16

    function TokenIndex(str::String, id::Integer)
        id < 0 && throw(DomainError("Token index must be > 0."))
        new(convert(String, str), convert(Int16, id))
    end

end

"""
    isless_tokens(first_token::TokenIndex, second_token::TokenIndex) -> Bool

Compare two `TokenIndex` objects by their string values.
It returns `true` if the first token's string is **lexicographically** less than the second's, and `false` otherwise.

# Examples
```jldoctest
julia> using Llama2;

julia> isless_tokens(TokenIndex("A", 1), TokenIndex("B", 2))
true

julia> isless_tokens(TokenIndex("B", 1), TokenIndex("A", 2))
false
```

"""
function isless_tokens(first_token::TokenIndex, second_token::TokenIndex)::Bool
    return isless(first_token.str, second_token.str)
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
- `vocab`: Token string sequences.  
- `vocab_scores`: Scores for each token.  
- `sorted_vocab`: Sorted token indices.  
- `vocab_size`: Number of vocabulary entries.  
- `max_token_length`: Maximum token length in bytes.  
- `byte_pieces`: Byte mapping (length 256).
"""
struct Tokenizer

    vocab::Vector{String}
    vocab_scores::Vector{Float32}
    sorted_vocab::Vector{TokenIndex}
    vocab_size::Int16
    max_token_length::UInt16
    byte_pieces::Vector{UInt8}

    function Tokenizer(vocab::AbstractVector{String},
        vocab_scores::AbstractVector{Float32},
        sorted_vocab::AbstractVector{TokenIndex},
        vocab_size::Integer,
        max_token_length::Integer,
        byte_pieces::AbstractVector{UInt8}
    )
        max_token_length < 0 && throw(DomainError("max_token_length must be > 0."))
        length(byte_pieces) != 256 && throw(ArgumentError("Length of byte_pieces must be 256."))
        
        new(convert(Vector{String}, vocab), 
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
    sorted_vocab = Vector{TokenIndex}(undef, vocab_size)

    vocab_scores = Vector{Float32}(undef, vocab_size)
    vocab = Vector{String}(undef, vocab_size)
    max_token_length = 0

    open(tokenizer_path, "r") do f
        max_token_length = Int(read(f, Int32))
        _ = read(f, 3624)
        for i in 1:vocab_size
            vocab_scores[i] = read(f, Float32)
            vocab_len = read(f, Int32)
            vocab[i] = String(read(f, vocab_len))
        end
    end

    Tokenizer(vocab, vocab_scores, sorted_vocab, vocab_size, max_token_length, byte_pieces)
end

"""
    str_lookup(str::String, sorted_vocab::Vector{TokenIndex}) -> Int16

Search for a given string `str` within a sorted vocabulary `sorted_vocab` of `TokenIndex` objects.
If the string is found, it returns the corresponding token ID;
**otherwise, it returns `-1`.** It uses a binary search for efficient lookup.

# Examples
```jldoctest
julia> using Llama2;

julia> str_lookup("aa", [TokenIndex("aa", 1), TokenIndex("bb", 2)])
1

julia> str_lookup("ba", [TokenIndex("aa", 1), TokenIndex("bb", 2)])
-1
```

"""
function str_lookup(str::String, sorted_vocab::Vector{TokenIndex})::Int16

    tok = TokenIndex(str, Int16(0))
    idx = searchsortedfirst(sorted_vocab, tok; lt = isless_tokens)
    if idx <= length(sorted_vocab) && sorted_vocab[idx].str == str
        return sorted_vocab[idx].id
    else
        return Int16(-1)
    end
end

"""
    encode

Converts a string `text` into a sequence of token IDs using a `Tokenizer`.
First ensure the tokenizer's vocabulary is sorted, then encode each character into its corresponding ID.
After that, iteratively merge token pairs with the highest scores to form longer tokens until no more merges are possible.
Return the final token ID sequence.

"""
function encode(tokenizer::Tokenizer, text::String)

    if !isassigned(tokenizer.sorted_vocab)
        for i in 1:tokenizer.vocab_size
            tokenizer.sorted_vocab[i] = TokenIndex(tokenizer.vocab[i], i)
        end
        sort!(tokenizer.sorted_vocab, lt = isless_tokens)
    end

    tokens = Vector{Integer}(undef, length(text))

    # Encode every input char with an id from vocab
    for (pos, char) in enumerate(string.(collect(text)))
        id = str_lookup(char, tokenizer.sorted_vocab)
        id == -1 && throw(ArgumentError("Can not encode prompt at position $pos"))
        tokens[pos] = id
    end

    while true
        best_score = -1.0f10
        best_id = -1
        best_idx = -1

        for i in 1:(length(tokens)-1)
            string = tokenizer.vocab[tokens[i]] * tokenizer.vocab[tokens[i+1]]
            id = str_lookup(string, tokenizer.sorted_vocab)
            if id != -1 && tokenizer.vocab_scores[id] > best_score
                best_score = tokenizer.vocab_scores[id]
                best_id = id
                best_idx = i
            end
        end

        # Can not find any more pairs to merge
        best_idx == -1 && break

        # Pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id

        deleteat!(tokens, best_idx+1)
    end

    return tokens
end
