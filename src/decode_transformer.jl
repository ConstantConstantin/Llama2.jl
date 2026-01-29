"""
    Transformer(path::String)

Load a binary file with location `path` and construct a `Transformer` from its content.
The file is expected to have a header of 7 Int32 values followed by Float32 data.

# Arguments
- `path::String`: Path to the binary model file.

# Returns
- `Transformer`: A transformer model ready for inference.

# Example
```
julia> t = Llama2.Transformer("/PATH/TO/YOUR.bin");
```
"""
function Transformer(path::String)

    open(path, "r") do data
        
        header = Vector{Int32}(undef, 7) # read header (7 Int32 values)
        read!(data, header) # read header into the vector
        config = Config(header...)

        shared_weights = config.vocab_size > 0
        head_size = div(config.dim, config.n_heads)
        n_layers = UInt64(config.n_layers)
        
        token_embedding_table = Matrix{Float32}(undef, config.dim, config.vocab_size)
        read!(data, token_embedding_table)
        token_embedding_table = permutedims(token_embedding_table)
        
        rms_att_weight = Matrix{Float32}(undef, config.dim, n_layers)
        read!(data, rms_att_weight)
        rms_att_weight = permutedims(rms_att_weight)
        
        wq = Array{Float32, 3}(undef, config.n_heads * head_size, config.dim, n_layers)
        read!(data, wq)
        wq = permutedims(wq, [3, 2, 1])
        
        wk = Array{Float32, 3}(undef, config.n_kv_heads * head_size, config.dim, n_layers)
        read!(data, wk)
        wk = permutedims(wk, [3, 2, 1])
        
        wv = Array{Float32, 3}(undef, config.n_kv_heads * head_size, config.dim, n_layers)
        read!(data, wv)
        wv = permutedims(wv, [3, 2, 1])
        
        wo = Array{Float32, 3}(undef, config.dim, config.n_heads * head_size, n_layers)
        read!(data, wo)
        wo = permutedims(wo, [3, 2, 1])
        
        rms_ffn_weight = Matrix{Float32}(undef, config.dim, n_layers)
        read!(data, rms_ffn_weight)
        rms_ffn_weight = permutedims(rms_ffn_weight)
        
        w1 = Array{Float32, 3}(undef, config.dim, config.hidden_dim, n_layers)
        read!(data, w1)
        w1 = permutedims(w1, [3, 2, 1])
        
        w2 = Array{Float32, 3}(undef, config.hidden_dim, config.dim, n_layers)
        read!(data, w2)
        w2 = permutedims(w2, [3, 2, 1])
        
        w3 = Array{Float32, 3}(undef, config.dim, config.hidden_dim, n_layers)
        read!(data, w3)
        w3 = permutedims(w3, [3, 2, 1])
        
        rms_final_weight = Vector{Float32}(undef, config.dim)
        read!(data, rms_final_weight)

        skip(data, 4 * config.seq_len * head_size)
        
        if shared_weights
            wcls = Matrix{Float32}(token_embedding_table)
        else
            wcls = Matrix{Float32}(undef, config.vocab_size, config.dim)
        end
        
        weights = TransformerWeights(token_embedding_table, rms_att_weight, rms_ffn_weight, wq, wk, wv, wo, w1, w2, w3, rms_final_weight, wcls)
        
        return Transformer(config, weights)
    end
    
end
