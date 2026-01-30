@testset "talk" begin
    
    using Random: seed!

    p = normpath(joinpath(@__DIR__, "..", "data", "stories15M.bin"))

    seed!(851216)

    @testset "talktollm" begin
        
        for (result, expectation) in ((talktollm(p), " Once upon a time, there was a boy named Tim. He lived in a small house with his mom, dad, and dog. Tim's house was in a big green park. One day, Tim and his dog went for a walk in the park. They saw many pretty flowers and big trees. Tim felt ready to play with his dog.\nAs they were walking, they found a bottle in the park. The bottle was big and red. Tim wanted to see what was inside. He opened it and found a bottle with a drink called wine. Tim was surprised! He didn't know what wine was.\nTim's dog showed Tim that wine was a drink. They never knew wine was a drink. Tim took a sip and it was sweet and cool. They were happy that they found something new and different in the park. Tim and his dog played in the park all day and had a great time."),
                                      (talktollm(p, "Some ducks on the pond "), "Some ducks on the pond 3 One had Jenny and Tim. But one day Jenny and Tim were playing near the pond. They started to be very noisy, squirting and chirping everywhere.\nFinally, Jenny's mom was calling them, \"Where are you going now? Come back so you can turn around?\"\nSo Jenny and Tim turned around again and saw that their mom was covered in ducks that had been bald. \nThe mom said, \"That's so interesting! Now you are more quiet\".\nJenna and Tim nodded and said, \"Yes!\""),
                                      (talktollm(p; max_tokens = 127), " Once upon a time, there was a little squirrel named Nutty. Nutty loved to collect nuts but one day, he found a black nut that he had never seen before. He was very proud of his black nut and showed it to all his friends.\nOne day, Nutty was climbing up a tree when he slipped and fell down. He hurt his leg and couldn't walk anymore. His friends came to help him and realized what had happened. They remembered that dogs can hear and keep them away from Nutty's big muscles.\nSadly, N"))

            @test result isa String
            @test result == expectation

        end

    end

    seed!(8512)

    @testset "chatwithllm" begin

        c = ChatBot(p)

        d = chatwithllm(c)
        e = chatwithllm(c, " a secret path")

        @test d isa String
        @test e isa String

        @test d == " Once upon a time, there was a little bird who loved singing. She flew everywhere in the forest, and sometimes even made up her own song for the other birds to sing along. One day, she met a little girl who was lost and couldn't find her way home. The little bird showed her"
        @test e == " a secret path that led the way to her home.\nThe little girl was grateful and thanked the bird. But as she started to leave, the little bird became selfish and scurried away. The little girl was sad and realized that she should have listened to the bird's advice and left the little"

    end

end
