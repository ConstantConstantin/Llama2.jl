"""
    talktollm(modelpath::String, [prompt::String]; max_tokens::Int, vocabpath::String, verbose::Bool)

Generate text using a pretrained LLama2 transformer model.
Return that text as a `String`.
Load the model from `modelpath` and the corresponding tokenizer from `vocabpath`
(which defaults to `"data/tokenizer.bin"`). Take an initial `prompt` `String`
to start the text generation and generate up to `max_tokens` tokens.
If `verbose`, print the text during generation.

```julia
julia> print(talktollm("/PATH/TO/YOUR/MODEL.bin"))
 Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her friends. One day, Lily was running and she fell and hit her head on a rock. She got a big ouchie and it started to bleed. 
Lily's mom took her to the doctor and the doctor said she needed a stitch. Lily was scared, but her mom was very dependable and told her they would be coming back home soon. 
After the doctor fixed Lily's knee, they went home and Lily's friends came to play again. But Lily's mom noticed that she was playing with a ball and some new toys. This made her very happy.

julia> print(talktollm("/PATH/TO/YOUR/MODEL.bin", "\"What is this?\""))
"What is this?" the woman asked.
The little girl looked at the bookion and said, "This is a book about a princess. Maybe we can use it together."
They decided to sit down and read the book together. They read about a beautiful garden with lovely flowers. The little girl loved the book very much and said, "I want to be a princess again!"
"Maybe, if you read me another book," the woman said.
From that day on, they would sit down and read the book every night before bed. They hoped that when they finished reading it, something magical would happen.
```
"""
function talktollm(modelpath::String, prompt::String = ""; max_tokens::Int=256, vocabpath::String = _vocabpath, verbose::Bool = false)

    transformer = Transformer(modelpath)
    tok = Tokenizer(vocabpath, transformer.config.vocab_size)

    input_tokens = encode(tok, prompt)

    result = Vector{Int32}()
    sizehint!(result, max_tokens)

    if isempty(input_tokens)
        input_tokens = [2] # default for empty prompt
    else
        push!(result, input_tokens[1])
        verbose &&  print(tok.vocab[input_tokens[1]])
    end

    token = input_tokens[1]
    n_input_tokens = length(input_tokens)

    for pos in 1:max_tokens

        logits = forward!(transformer, Int32(token), Int32(pos))

        if pos < n_input_tokens
            next = input_tokens[pos + 1]
        else
            softmax!(logits)
            next = wsample(logits)
        end

        next == 2 && break

        push!(result, next)
        verbose && print(tok.vocab[next])

        token = next
        
    end

    verbose && println()
    
    return string(broadcast(x -> tok.vocab[x], result)...)
end

"""
    chatwithllm(bot::ChatBot, [prompt::String]; max_tokens::Int, verbose::Bool)

Generate text using a pretrained LLama2 transformer model.
Return that text as a `String`.

Multiple calls on the same instance of `ChatBot` respect the previously generated tokens and continue generation from there.
Take an initial `prompt` `String`
to start the text generation and generate up to `max_tokens` tokens.
If `verbose`, print the text during generation.

```julia
julia> c = ChatBot("data/stories15M.bin");

julia> print(chatwithllm(c); max_tokens = 63)
 Once upon a time, there was an old house with an ancient sign inside. The sign was very big and had many words on it. One day, a little girl went to visit the old house. She wanted to see what was inside.
The old house said, "Hello? Can I come in?"

julia> print(chatwithllm(c, "\nThe little girl said:"; max_tokens = 63))

The little girl said: "Yes please! Can I come in too?"
The old house thought for moments before it said, "Yes. This light is available for you 30 cent a nightmare."
The little girl was very excited. She said thank you and then, followed her favorite sign

julia> print(chatwithllm(c, "until"; max_tokens = 63))
until she saw there was a beautiful light online.
When the old house passed, the girl happily went inside. It was very old, but it had been there for a long time. The old house was very special, and she thought the light was the prettiest thing ever.
```
"""
function chatwithllm(bot::ChatBot, prompt::String = ""; max_tokens::Int = 256, verbose::Bool = false)

    transformer = bot.transformer
    tok = bot.tokenizer
    
    input_tokens = encode(tok, prompt)

    result = Vector{Int32}()
    sizehint!(result, max_tokens)

    if isempty(input_tokens)
        input_tokens = [bot.last_token] # default for empty prompt
    else
        push!(result, input_tokens[1])
        verbose && print(tok.vocab[input_tokens[1]])
    end

    token = input_tokens[1]
    n_input_tokens = length(input_tokens)

    for pos in bot.pos:(bot.pos + max_tokens)
        if pos >= transformer.config.seq_len
            bot.pos = pos
            @info "YOUR CHAT REACHED MAXIMUM SEQUENCE LENGTH!"
            break
        end
        logits = forward!(transformer, Int32(token), Int32(pos))

        if pos + 1 - bot.pos < n_input_tokens
            next = input_tokens[pos + 2 - bot.pos]
        else
            softmax!(logits)
            next = wsample(logits)
        end

        if next == 2
            break
        end

        push!(result, next)
        verbose && print(tok.vocab[next])

        token = next
        
    end

    bot.pos += length(result)
    !isempty(result) && (bot.last_token = result[end])

    verbose && println()
    
    return string(broadcast(x -> tok.vocab[x], result)...)
end
