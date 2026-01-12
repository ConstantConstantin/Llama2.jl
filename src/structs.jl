"""
    Config

Create a `Config` containing 7 `Int32`. These describe meta-data to read values from an input file.

# Developer Notes
This is an internal struct.
"""
struct Config
    dim::Int32
    hidden_dim::Int32
    n_layers::Int32
    n_heads::Int32
    n_kv_heads::Int32
    vocab_size::Int32
    seq_len::Int32
end

"""
    TransformerWeights

Create a `TransformerWeights` containing several `Float32` containers. These describe actual weight data that is loaded from an input file.

# Developer Notes
This is an internal struct.
"""
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

"""
    RunState

Create a `RunState` containing several `Float32` containers. These reflect the state of the `Transformer` at run-time.

# Developer Notes
This is an internal struct.
"""
mutable struct RunState
    x::Vector{Float32}
    xb2::Vector{Float32}
    hb::Vector{Float32}
    hb2::Vector{Float32}
    q::Vector{Float32}
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

    """
    Transformer(config::Config, weights::TransformerWeights)

Create a `Transformer` with data from `config` and `weights`. The `RunState` containers are initialized empty and just are assigned the corresponding dimensions.

# Developer Notes
This is an internal struct.
     """
    function Transformer(config::Config, weights::TransformerWeights)
        kv_dim = div((config.dim * config.n_kv_heads), config.n_heads)
        x = Vector{Float32}(undef, config.dim)
        xb2 = Vector{Float32}(undef, config.dim)
        hb = Vector{Float32}(undef, config.hidden_dim)
        hb2 = Vector{Float32}(undef, config.hidden_dim)
        q = Vector{Float32}(undef, config.dim)
        key_cache = Array{Float32, 3}(undef, config.n_layers, config.seq_len, kv_dim)
        value_cache = Array{Float32, 3}(undef, config.n_layers, config.seq_len, kv_dim)
        att = Matrix{Float32}(undef, config.n_heads, config.seq_len)
        logits = Vector{Float32}(undef, config.vocab_size)
        new(config, weights, RunState(x, xb2, hb, hb2, q, att, logits, key_cache, value_cache))
    end
end
