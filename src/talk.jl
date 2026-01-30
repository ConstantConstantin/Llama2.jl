const _vocabpath = normpath(joinpath(@__DIR__, "..", "data", "tokenizer.bin"))

"""
    LlamaChat

A stateful chat session with a Llama2 model that accumulates conversation history.

# Fields
- `transformer::Transformer`: The loaded transformer model
- `tokenizer::Tokenizer`: The tokenizer for encoding/decoding
- `history::String`: Accumulated conversation history
- `pos::Int`: Current token position in the sequence
- `last_token::Int32`: Last generated token (for continuing generation)
"""
mutable struct LlamaChat
    transformer::Transformer
    tokenizer::Tokenizer
    history::String
    pos::Int
    last_token::Int32
end

"""
    LlamaChat(modelpath::String; vocabpath::String)

Initialize a new chat session with a Llama2 model.

# Examples
```julia
chat = LlamaChat("/path/to/model.bin")
```
"""
function LlamaChat(modelpath::String; vocabpath::String = _vocabpath)
    transformer = Transformer(modelpath)
    tokenizer = Tokenizer(vocabpath, transformer.config.vocab_size)
    return LlamaChat(transformer, tokenizer, "", 0, Int32(0))
end

"""
    talk!(chat::LlamaChat, prompt::String; max_tokens::Int=255, verbose::Bool=true)

Continue conversation with an existing chat session. The prompt is appended to
the conversation history and generation continues from the current position.

# Examples
```julia
chat = LlamaChat("/path/to/model.bin")
talk!(chat, "Hello, who are you?")
talk!(chat, " Tell me more")  # Continues with context
```
"""
function talk!(chat::LlamaChat, prompt::String; max_tokens::Int=255, verbose::Bool=true)
    chat.history *= prompt
    
    input_tokens = encode(chat.tokenizer, chat.history)
    
    if isempty(input_tokens)
        input_tokens = [Int32(2)]  
    end
    
    n_input_tokens = length(input_tokens)
    start_pos = chat.pos == 0 ? 1 : chat.pos + 1
    
    token = start_pos <= n_input_tokens ? input_tokens[start_pos] : chat.last_token
    
    result = String[]
    
    for pos in start_pos:(start_pos + max_tokens - 1)
        logits = forward!(chat.transformer, Int32(token), Int32(pos))
        
        if pos < n_input_tokens
            next = input_tokens[pos + 1]
        else
            softmax!(logits)
            next = wsample(logits)
        end
        
        next == 2 && break
        
        if pos >= n_input_tokens
            token_str = chat.tokenizer.vocab[next]
            verbose && print(token_str)
            push!(result, token_str)
        end
        
        chat.pos = pos
        chat.last_token = Int32(next)
        token = next
    end
    
    generated = join(result)
    chat.history *= generated
    
    verbose && println()
    return generated
end

