@testset "talk" begin

    @testset "talktollm" begin

        p = normpath(joinpath(@__DIR__, "..", "data", "stories15M.bin"))
        
        for result in (talktollm(p), talktollm(p, "Some ducks on the pond "), talktollm(p; max_tokens = 127))

            @test result isa String

        end

    end

end
