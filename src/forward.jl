"""
    rmsnorm(x, w)

Calculate the rmsnorm of `x` and `w`, the scaled product 'λw * x'.

# Examples
```jldoctest
julia>  x = [1.0f0,2,3];

julia>  w = [1.0f0,1,1];

julia> o = Llama2.rmsnorm(x, w) 
3-element Vector{Float32}:
 0.46290955
 0.9258191
 1.3887286
```
"""
function rmsnorm(x::AbstractVector{Float32}, w::AbstractVector{Float32})

    (length(w) != length(x)) && throw(DimensionMismatch("x and w must have the same dimensions"))
    isempty(x) && throw(ArgumentError("x must not be empty"))

    #calculate sum of squares
    ss = dot(x, x)

    ss = ss / length(x) + 1f-5
    scale = inv(sqrt(ss))

    return scale * w .* x
end


"""
softmax!(x)
Updates the Output of an Layer 'x' with the softmax of the input.

# Examples
```jldoctest
julia> x = [-1.0f0,0,1];

julia> Llama2.softmax!(x);

julia> x
3-element Vector{Float32}:
 0.09003057
 0.24472848
 0.66524094
```
"""
function softmax!(x::AbstractVector{Float32})

    isempty(x) && throw(ArgumentError("x must not be empty"))

    max_x = maximum(x)

    for i in eachindex(x)
        x[i] = exp(x[i] - max_x)
    end

    x ./= sum(x)

    return x
end

function forward!(transformer::Transformer, token::Int32, pos::Int32)

    config = transformer.config
    weights = transformer.weights
    state = transformer.state

    dim = config.dim
    kv_dim = div(dim * config.n_kv_heads, config.n_heads)
    kv_mul = div(config.n_heads, config.n_kv_heads)
    hidden_dim = config.hidden_dim
    head_size = div(dim, config.n_heads)

    # assigning input token embedding to x
    x = weights.token_embedding_table[token, :]
    
    for l in 1:config.n_layers

        xb = rmsnorm(x, weights.rms_att_weight[l, :])

        k = @view state.key_cache[l, pos, :]
        v = @view state.value_cache[l, pos, :]
        # matmul to get q, k, v

        q = weights.wq[l, :, :] * xb
        k .= weights.wk[l, :, :] * xb
        v .= weights.wv[l, :, :] * xb

        for i in 1:2:dim

            head_dim = (i - 1) % head_size
            val = (pos - 1)  / (10000.0f0^(head_dim / head_size))
            fcr = cos(val)
            fci = sin(val)
            
            for vi in 1:(1 + (i <= kv_dim))
                if vi == 1
                    vec = @view q[i:(i + 1)]
                else
                    vec = @view k[i:(i + 1)]
                end
                v0 = vec[1]
                v1 = vec[2]
                vec[1] = v0 * fcr - v1 * fci
                vec[2] = v0 * fci + v1 * fcr
            end

        end

        xb .= 0

        for h in 1:config.n_heads # multi-head attention

            q_head = @view q[((h - 1) * head_size + 1):(h  * head_size)]
            att = Vector{Float32}(undef, pos)

            for t in 1:pos

                k = state.key_cache[l, t, (div(h - 1, kv_mul) * head_size + 1):((div(h - 1, kv_mul) + 1) * head_size)]

                att[t] = dot(q_head, k)/sqrt(head_size)

            end

            softmax!(att)

            xb_head = @view xb[((h - 1) * head_size + 1):(h * head_size)]

            for t in 1:pos

                v = state.value_cache[l, t, (div(h - 1, kv_mul) * head_size + 1):((div(h - 1, kv_mul) + 1) * head_size)]
                
                xb_head .+= att[t] * v

            end

        end

        x .+= weights.wo[l, :, :] * xb

        xb = rmsnorm(x, weights.rms_ffn_weight[l, :])

        hb = weights.w1[l, :, :] * xb
        hb2 = weights.w3[l, :, :] * xb

        for i in 1:hidden_dim
            hb[i] = hb[i] * hb2[i] / (1.0f0 + exp(-hb[i]))
        end

        x .+= weights.w2[l, :, :] * hb

    end
    
    x .= rmsnorm(x, weights.rms_final_weight) # final rmsnorm
    
    # classifier into logits
    state.logits .= weights.wcls * x
    
    return state.logits

end
