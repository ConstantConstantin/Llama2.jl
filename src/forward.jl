using LoopVectorization: @turbo


"""
rmsnorm!(o, x, w)
Updates the Output of an Layer 'o' with the rmsnorm scaled weights and inputs 'w .* x'.
Using @turbo for performance optimization.

# Examples
```jldoctest
julia> using Llama2;

julia>   x = [1.0f0,2,3];
julia>   w = [1.0f0,1,1];
julia>   o = [0.0f0,0,0];

julia> rmsnorm!(o, x, w) 

julia> o
3-element Vector{Float32}:
 0.46290955
 0.9258191
 1.3887286

```

# Developer Notes
Dimensions of o, x, and w not quite sure yet.

"""

function rmsnorm!(o::Vector{T}, x::Vector{T}, w::Vector{T}) where {T<:Float32}

    @assert length(w) == length(o) == length(x) "x, o, and w must have the same dimensions"
    @assert !isempty(x) "x must not be empty"

    ss = 0.0

    #calculate sum of squares

    @turbo for i in eachindex(x)
        ss += x[i] * x[i]
    end


    ss = ss / length(x) + 1e-5
    scale = inv(sqrt(ss))

    #normalize and scale 

    @turbo for i in eachindex(x)
        o[i] = w[i] * scale * x[i]
    end

    return nothing
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

    @assert !isempty(x) "x must not be empty"

    max_x = maximum(x)

    @turbo for i in eachindex(x)
        x[i] = exp(x[i] - max_x)
    end

    norm = inv(sum(x))

    @turbo for i in eachindex(x)
        x[i] *= norm
    end

    return nothing
end

function forward!(transformer::Transformer)
    for i in eachindex(transformer.config.n_layers)
        
    end
end



