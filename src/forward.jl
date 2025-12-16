using LoopVectorization: @turbo

function rmsnorm!(o::AbstractVector{T}, x::AbstractVector{T}, w::AbstractVector{T}) where {T<:AbstractFloat}
    """
    rmsnorm!(o, x, w)
    Updates the Output of an Layer 'o' with the rmsnorm scaled weights and inputs 'w .* x'.
    Using @turbo for performance optimization.

    # Examples
    ```jldoctest
    julia> using Llama2;

    julia> x = rand(Float32, 24)
           w = rand(Float32, 24)
           o = similar(x)

    julia> rmsnorm!(o, x, w)
    ```

    # Developer Notes
    Dimensions of o, x, and w not quite sure yet.

    """

    @assert length(o) == length(x) == length(w) "x, o, and w must have the same length"
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



function softmax!(x::AbstractVector{T}) where T<:AbstractFloat
    """
    softmax!(x)
    Updates the Output of an Layer 'x' with the softmax of the input.
    Using @turbo for performance optimization.

    # Examples
    ```jldoctest
    julia> using Llama2;

    julia> x = rand(Float32, 24)

    julia> softmax!(x)
    ```

    """
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

