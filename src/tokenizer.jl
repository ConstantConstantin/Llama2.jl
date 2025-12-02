struct TokenIndex{VI<:AbstractVector{UInt8}, I<:Integer}
    str::VI
    id::I

    function TokenIndex(str::VI, id::I) where {VI<:AbstractVector{UInt8}, I<:Integer}
        id < 0 && error("Token index must be > 0.")
        new{VI, I}(str, id)
    end

end

function compare_tokens(first_token::TokenIndex, second_token::TokenIndex)::Bool
    return (first_token.str == second_token.str)
end

struct Tokenizer{
    VVI<:AbstractVector{AbstractVector{UInt8}}, 
    VF<:AbstractVector{Float32},  
    VI<:AbstractVector{UInt8}, 
    I<:Integer,
    J<:Integer,
    VTI<:AbstractVector{TokenIndex{VI,I}}
}
    vocab::VVI
    vocab_scores::VF
    sorted_vocab::VTI
    vocab_size::I
    max_token_length::J
    byte_pieces::VI

    function Tokenizer(vocab::VVI, 
        vocab_scores::VF, 
        sorted_vocab::VTI, 
        vocab_size::I, 
        max_token_length::J, 
        byte_pieces::VI
    ) where {
        VVI<:AbstractVector{AbstractVector{UInt8}}, 
        VF<:AbstractVector{Float32}, 
        VI<:AbstractVector{UInt8}, 
        I<:Integer,
        J<:Integer,
        VTI<:AbstractVector{TokenIndex{VI,I}}
    }
        max_token_length < 0 && error("max_token_length must be > 0.")
        length(byte_pieces) != 256 && error("Length of byte_pieces must be 256.")
        
        new{VVI, VF, VTI, VI, I, J}(vocab, vocab_scores, sorted_vocab, vocab_size, max_token_length, byte_pieces)
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
