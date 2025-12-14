struct Config
    dim::Int32
    hidden_dim::Int32
    n_layers::Int32
    n_heads::Int32
    n_kv_heads::Int32
    vocab_size::Int32
    seq_len::Int32
end

struct TransformerWeights
    token_embedding_table::Matrix{Float32}
    rms_att_weight::Matrix{Float32}
    rms_ffn_weight::Matrix{Float32}
    wq::Array{Float32, 3}
    wk::Array{Float32, 3}
    wv::Array{Float32, 3}
    wo::Array{Float32, 3}
    w1::Array{Float32, 3}
    w2::Array{Float32, 3}
    w3::Array{Float32, 3}
    rms_final_weight::Vector{Float32}
    wcls::Matrix{Float32}
end

struct RunState
    x::Vector{Float32}
    xb::Vector{Float32}
    xb2::Vector{Float32}
    hb::Vector{Float32}
    hb2::Vector{Float32}
    q::Vector{Float32}
    k::Vector{Float32}
    v::Vector{Float32}
    att::Matrix{Float32}
    logits::Vector{Float32}
    key_cache::Array{Float32, 3}
    value_cache::Array{Float32, 3}
end

struct Transformer
    config::Config
    weights::TransformerWeights
    state::RunState
    # omitting the fd/file size info for now because the content already should be stored into the fields of config and weights

    function Transformer(config::Config, weights::TransformerWeights)
        kv_dim = div((config.dim * config.n_kv_heads), config.n_heads)
        x = Vector{Float32}(undef, config.dim)
        xb = Vector{Float32}(undef, config.dim)
        xb2 = Vector{Float32}(undef, config.dim)
        hb = Vector{Float32}(undef, config.hidden_dim)
        hb2 = Vector{Float32}(undef, config.hidden_dim)
        q = Vector{Float32}(undef, config.dim)
        k = Vector{Float32}(undef, config.dim)
        v = Vector{Float32}(undef, config.dim)
        key_cache = Array{Float32, 3}(undef, config.n_layers, config.seq_len, kv_dim)
        value_cache = Array{Float32, 3}(undef, config.n_layers, config.seq_len, kv_dim)
        att = Matrix{Float32}(undef, config.n_heads, config.seq_len)
        logits = Vector{Float32}(undef, config.vocab_size)
        new(config, weights, RunState(x, xb, xb2, hb, hb2, q, k, v, att, logits, key_cache, value_cache))
    end
end
