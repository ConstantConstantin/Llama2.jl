using Random

"""
    ProbIndex

Create a `ProbIndex` from an `AbstractFloat` and an `Integer`.

`prob` contains a proberbility and is stored as an `Int32`
and `index` is stored as a `Float32`.
Throw a `DomainError` if `index < 0`.


# Examples
```jldoctest
julia> using Llama2;

julia> ProbIndex(1.0, 2)
ProbIndex(1.0f0, 2)

julia> ProbIndex(1.0, -1)
ERROR: DomainError with Prob index must be > 0.:
[...]
```
"""
struct ProbIndex
    prob::Float32
    index::Int32

    function ProbIndex(prob::AbstractFloat, index::Integer)
        index < 0 && throw(DomainError("Prob index must be > 0."))
        new(convert(Float32, prob), convert(Int32, index))
    end
end

"""
    Sampler

Stateful sampler for converting model logits into token indices.

Encapsulate sampling configuration (temperature, nucleus sampling)
and internal buffers needed for efficient token sampling.

# Constructors
- `Sampler(vocab_size::Int32, temperature::Float32, topp::Float32, rng_seed::Int128)`  
  Construct a sampler with automatically allocated internal buffers for
  nucleus (top-p) sampling.

- `Sampler(vocab_size::Int32, probindex::Vector{ProbIndex},
            temperature::Float32, topp::Float32, rng_state::Int128)`  
  Construct a sampler using a caller-provided `probindex` workspace buffer.
  The buffer must have length at least `vocab_size`.

# Fields
- `vocab_size::Int32`: Vocabulary size
- `temperature::Float32`: Sampling temperature (0 = greedy)
- `topp::Float32`: Nucleus sampling threshold
- `rng_state::Int128`: Random number generator state

`Sampler` is *callable* and can be applied to a vector of logits to obtain
the next token index.
"""
struct Sampler
    vocab_size::Int32
    probindex::Vector{ProbIndex} # karpathy: buffer used in top-p sampling
    temperature::Float32
    topp::Float32
    rng_state::Int128

    function Sampler(vocab_size::Integer,
        probindex::Vector{ProbIndex},
        temperature::AbstractFloat,
        topp::AbstractFloat,
        rng_state::Integer
    )
        vocab_size <= 0 && throw(DomainError("vocab_size must be > 0."))
        new(convert(Int32, vocab_size),
            convert(Vector{ProbIndex}, probindex),
            convert(Float32, temperature),
            convert(Float32, topp),
            convert(Int128, rng_state)
        )
    end
end
function Sampler(vocab_size::Integer, temperature::AbstractFloat, topp::AbstractFloat, rng_seed::Integer) 
    # karpathy: buffer only used with nucleus sampling; may not need but it's ~small:
    probindex = Vector{ProbIndex}(undef, vocab_size)
    Sampler(vocab_size, probindex, temperature, topp, rng_seed)
end

"""
    sample_mult(probabilities, coin) -> Int

Sample an index from a multinomial distribution.

Given a vector of normalized probabilities and a uniform random number
`coin ∈ [0, 1)`, returns the first index whose cumulative probability
exceeds `coin`.

# Arguments
- `probabilities::Vector{Float32}`: Probability mass function (must sum to 1).
- `coin::Float64`: Uniform random number in `[0, 1)`.

# Returns
- Index of the sampled element.

If numerical roundoff prevents an early return, the last index is returned.
"""
function sample_mult(probabilities::Vector{Float32}, coin::Float64)
    cdf = 0.0f0
    for i in eachindex(probabilities)
        cdf += probabilities[i]
        coin < cdf && return i
    end
    return lastindex(probabilities)
end

"""
    isless_probindex(a::ProbIndex, b::ProbIndex) -> Bool

Comparison function for ordering `ProbIndex` values by probability.

Return `true` if `a.prob < b.prob`. Intended for use as the `lt` argument
to sorting routines.
"""
function isless_probindex(first_probindex::ProbIndex, second_probindex::ProbIndex)
    return first_probindex.prob < second_probindex.prob
end

"""
    sample_topp(probabilities, topp, probindex, coin) -> Int

Sample an index using nucleus (top-p) sampling.

Selects the smallest set of tokens whose cumulative probability mass exceeds
`topp`, then samples from this restricted distribution using the provided
random number.

# Arguments
- `probabilities::Vector{Float32}`: Normalized probability distribution.
- `topp::Float32`: Cumulative probability threshold (`0 < topp < 1`).
- `probindex::Vector{ProbIndex}`: Preallocated workspace for sorting and
  indexing candidate tokens.
- `coin::Float64`: Uniform random number in `[0, 1)`.

# Returns
- Index of the sampled token.

The `probindex` buffer is mutated and reused to avoid allocations.
"""
function sample_topp(probabilities::Vector{Float32},
    topp::Float32,
    probindex::Vector{ProbIndex},
    coin::Float64
    )

    cutoff = (1.0f0 - topp) / length(probabilities)
    probindex.index = findall(x -> x >= cutoff, probabilities)
    probindex.prob = probabilities[probindex.index]

    sort!(probindex, lt=isless_probindex)

    cumulative_prob = 0.0f0
    last_idx = lastindex(probindex)
    @inbounds for i in eachindex(probindex)
        cumulative_prob += probindex[i].prob
        if cumulative_prob > topp
            last_idx = i
            break
        end
    end

    r = coin * cumulative_prob
    cdf = 0.0f0
    @inbounds for i in eachindex(probindex)
        cdf += probindex[i].prob
        if r < cdf
            return probindex[i].index
        end
    end
    return probindex[last_idx].index
end

"""
    (sampler::Sampler)(logits::Vector{Float32}) -> Int

Sample a token index from model logits.

If `sampler.temperature == 0`, perform greedy argmax decoding.
Otherwise, apply temperature scaling and sample according to:

- multinomial sampling if `topp ≤ 0` or `topp ≥ 1`
- nucleus (top-p) sampling if `0 < topp < 1`

The sampler is stateful and uses its internal RNG state.

# Arguments
- `logits`: Unnormalized model logits (length = `vocab_size`)

# Returns
- Token index (`Int`)
"""
function (sampler::Sampler)(logits::Vector{Float32})
    if sampler.temperature == 0.0
        next = argmax(logits)
    else
        logits = logits / sampler.temperature
        softmax!(logits)

        rng = MersenneTwister(sampler.rng_state)
        coin = rand(rng)

        if sampler.topp <= 0 || sampler.topp >= 1
            next = sample_mult(logits, coin)
        else
            next = sample_topp(logits, sampler.topp, sampler.probindex, coin)
        end
    end

    return next
end
