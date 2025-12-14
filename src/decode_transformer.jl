"""
    decode_transformer(path::String)

Load a binary file with location `path` and construct a `Transformer` from its content.
The file is expected to have a header of 7 Int32 values followed by Float32 data.

# Example
```
julia> t = Llama2.decode_transformer("/PATH/TO/YOUR.bin");
```
"""
function decode_transformer(path::String)

    open(path, "r") do data
        
        header = Vector{Int32}(undef, 7) # read header (7 Int32 values)
        read!(data, header) # read header into the vector
        config = Config(header...)
        
        shared_weights = config.vocab_size > 0
        head_size = div(config.dim, config.n_heads)
        n_layers = UInt64(config.n_layers)
        
        token_embedding_table = Matrix{Float32}(undef, config.vocab_size, config.dim)
        read!(data, token_embedding_table)
        
        rms_att_weight = Matrix{Float32}(undef, n_layers, config.dim)
        read!(data, rms_att_weight)
        
        wq = Array{Float32, 3}(undef, n_layers, config.dim, config.n_heads * head_size)
        read!(data, wq)
        
        wk = Array{Float32, 3}(undef, n_layers, config.dim, config.n_kv_heads * head_size)
        read!(data, wk)
        
        wv = Array{Float32, 3}(undef, n_layers, config.dim, config.n_kv_heads * head_size)
        read!(data, wv)
        
        wo = Array{Float32}(undef, n_layers, config.n_heads * head_size, config.dim)
        read!(data, wo)
        
        rms_ffn_weight = Matrix{Float32}(undef, n_layers, config.dim)
        read!(data, rms_ffn_weight)
        
        w1 = Array{Float32, 3}(undef, n_layers, config.dim, config.hidden_dim)
        read!(data, w1)
        
        w2 = Array{Float32, 3}(undef, n_layers, config.hidden_dim, config.dim)
        read!(data, w2)
        
        w3 = Array{Float32, 3}(undef, n_layers, config.dim, config.hidden_dim)
        read!(data, w3)
        
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
