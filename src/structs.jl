struct Config
    dim::Int
    hidden_dim::Int
    n_layers::Int
    n_heads::Int
    n_kv_heads::Int
    vocab_size::Int
    seq_len::Int
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
    wcls::Vector{Float32}
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
    fd::Int
    data::Vector{Float32}
    file_size::Int
end
