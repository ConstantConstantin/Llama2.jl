using LoopVectorization: @turbo

"""
    rmsnorm(x, w)

Calculate the rmsnorm of `x` and `w`, the scaled product 'λw * x'.

# Examples
```jldoctest
julia> using Llama2;

julia>   x = [1.0f0,2,3];
julia>   w = [1.0f0,1,1];

julia> o = rmsnorm(x, w) 
3-element Vector{Float32}:
 0.46290955
 0.9258191
 1.3887286

```

# Developer Notes
Dimensions of o, x, and w not quite sure yet.
"""
function rmsnorm(x::Vector{Float32}, w::Vector{Float32})

    (length(w) != length(x)) && throw(DimensionMismatch("x and w must have the same dimensions"))
    isempty(x) && throw(ArgumentError("x must not be empty"))

    #calculate sum of squares
    ss = 0.0f0
    ss += sum(x .^ 2)

    ss = ss / length(x) + 1e-5
    scale = inv(sqrt(ss))

    return scale * w .* x
end


"""
softmax!(x)
Updates the Output of an Layer 'x' with the softmax of the input.
Using @turbo for performance optimization.

# Examples
```jldoctest
julia> using Llama2;

julia> x = [-1.0f0,0,1];

julia> softmax!(x)

julia> x
3-element Vector{Float32}:
 0.09003057
 0.24472845
 0.66524094
```
"""
function softmax!(x::Vector{Float32})

    isempty(x) && throw(ArgumentError("x must not be empty"))

    max_x = maximum(x)

    @turbo for i in eachindex(x)
        x[i] = exp(x[i] - max_x)
    end

    norm = inv(sum(x))

    x *= norm

    return nothing
end

function forward!(transformer::Transformer, token::Int32, pos::Int32)

    config = transformer.config
    weights = transformer.weights
    state = transformer.state

    x = state.x
    dim = config.dim
    kv_dim = div(config.dim * config.n_kv_heads, config.n_heads)
    kv_mul = div(config.n_heads, config.n_kv_heads)
    hidden_dim = config.hidden_dim
    head_size = div(dim, config.n_heads)
    seq_len = config.seq_len

    # assigning input token embedding to x
    x .= weights.token_embedding_table[token, :]

    for l in 1:config.n_layers
        print(config.n_layers)
        xb = rmsnorm(x, weights.rms_att_weight[l, :])

        loff = l * seq_len * kv_dim
        k = @view state.key_cache[l, pos * kv_dim + 1, :]
        v = @view state.value_cache[l, pos * kv_dim + 1, :]
        # matmul to get q, k, v

        state.q = weights.wq[l, :, :] * xb
        k .= weights.wk[l * dim * kv_dim] * xb
        v .= weights.wv[l * dim * kv_dim] * xb

        for i in 1:2:dim

            head_dim = i % head_size
            freq = 1.0f0 / (10000.0f0^(head_dim / head_size))
            val = pos * freq
            fcr = cos(val)
            fci = sin(val)
            
            for v in 1:(1 + (i < kv_dim))
                vec = v == 0 ? state.q : k
                v0 = vec[i]
                v1 = vec[i + 1]
                vec[i] = v0 * fcr - v1 * fci
                vec[i + 1] = v0 * fci + v1 * fcr
            end

        end
        
        for h in 1:config.n_heads # multi-head attention

            q = state.q[(h * head_size):((h + 1) * head_size)]
            att = @view state.att[h, :]

            for t in 1:(pos + 1)

                k = state.key_cache[loff, t * kv_dim, (div(h, kv_mul) * head_size):((div(h, kv_mul) + 1) * head_size)]

                score = dot(q, k)/sqrt(head_size)
                att[t] = score

            end

            softmax!(att, pos + 1)

            xb_head = @view xb[(h * head_size):((h + 1) * head_size)]

            for t in 1:(pos + 1)

                v = state.value_cache[loff, t * kv_dim, (div(h, kv_mul) * head_size):((div(h, kv_mul) + 1) * head_size)]
                a = att[t]
                
                xb_head += a * v

            end

        end

        state.xb2 = wo[l * dim * dim, :, :] * state.xb

        x += state.xb2

        xb = rmsnorm(x, weights.rms_ffn_weight[l, :])

        state.hb = weights.w1[l * dim * hidden_dim, :, :] * state.xb
        state.hb2 = weights.w3[l * dim * hidden_dim, :, :] * state.xb

        for i in 1:hidden_dim
            val = state.hb[i]
            val *= (1.0f / (1.0f + exp(-val)))
            val *= state.hb2[i]
            state.hb[i] = val
        end

        state.xb = weights.w2[l * dim * hidden_dim, :, :] * state.hb

        x += state.xb
        
    end

    x .= rmsnorm(x, weights.rms_final_weight) # final rmsnorm
    
    # classifier into logits
    state.logits = weights.wcls * x
    
    return state.logits

end
