@testset "talk" begin
    
    using Random: seed!

    p = normpath(joinpath(@__DIR__, "..", "data", "stories15M.bin"))

    seed!(851216)

    @testset "talktollm" begin
        
        for (result, expectation) in ((talktollm(p), " Once upon a time, there was a boy named Tim. He lived in a small house with his mom, dad, and dog. Tim's house was in a big green park. One day, Tim and his dog went for a walk in the park. They saw many pretty flowers and big trees. Tim felt ready to play with his dog.\nAs they were walking, they found a bottle in the park. The bottle was big and red. Tim wanted to see what was inside. He opened it and found a bottle with a drink called wine. Tim was surprised! He didn't know what wine was.\nTim's dog showed Tim that wine was a drink. They never knew wine was a drink. Tim took a sip and it was sweet and cool. They were happy that they found something new and different in the park. Tim and his dog played in the park all day and had a great time."),
                                      (talktollm(p, "Some ducks on the pond "), "Some ducks on the pond 3 One had Jenny and Tim. But one day Jenny and Tim were playing near the pond. They started to be very noisy, squirting and chirping everywhere.\nFinally, Jenny's mom was calling them, \"Where are you going now? Come back so you can turn around?\"\nSo Jenny and Tim turned around again and saw that their mom was covered in ducks that had been bald. \nThe mom said, \"That's so interesting! Now you are more quiet\".\nJenna and Tim nodded and said, \"Yes!\""),
                                      (talktollm(p; max_tokens = 127), " Once upon a time, there was a little squirrel named Nutty. Nutty loved to collect nuts but one day, he found a black nut that he had never seen before. He was very proud of his black nut and showed it to all his friends.\nOne day, Nutty was climbing up a tree when he slipped and fell down. He hurt his leg and couldn't walk anymore. His friends came to help him and realized what had happened. They remembered that dogs can hear and keep them away from Nutty's big muscles.\nSadly, N"),
                                      (talktollm(p; temperature=0.f0), " Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.\nLily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to knock the ball down.\nLily found a stick and tried to hit the ball. But the stick was too short. She tried again and again, but she couldn't reach it. She felt sad.\nSuddenly, a kind man came by and saw Lily. He asked her what was wrong. Lily told him about the ball. The man smiled and said, \"I have a useful idea!\" He took out a long stick and used it to knock the ball down. Lily was so happy! She thanked the man and they played together in the sunshine."),
                                      (talktollm(p; temperature=1.f0), " Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big cloud in the sky. It looked like a real dragon! She wanted to touch it, but her mom said it was too far away.\nLily didn't listen and ran towards the cloud. She felt helpless and scared. Suddenly, the cloud turned into a big dragon! The dragon used its sharp claws to cut through the sky. Lily was amazed and happy.\nBut then, the dragon started to breathe fire. The dragon started to fight with Lily. She tried to run away, but the dragon was too big. In the end, the dragon took Lily away and she was never seen again."),
                                      (talktollm(p; temperature=1.2f0, topp=0.5f0), " Once upon a time, there was a fish named Bob. Bob was a proud fish. He lived in a big tank with many other fish. One day, a new fish came to the tank. This fish was very nice and had many colors. Bob and the new fish played together every day.\nOne day, Bob had a very long goo in his tank. He had a different color. This make him shake because it was the biggest swimming shower. The other fish liked to see it fall on the side of the tank. But Bob's own tank was small and rough. The other fish could not go there, and the good part of the tank had broken down.\nBob's mean, disgusting attitude made the other fish mad. They didn't want to play with Bob anymore. So, the shiny inperon won the swim show. All the fish stayed at the finish line. The special color started to come out of the pool. Bob saw this and realized that the friendly fish, no matter how it shook, changed it to clear their harsh talk. He changed his funny taste into a soft slap and let the other fish live. In the end, everyone had a happy t"))

            @test result isa String
            @test result == expectation

        end

    end

    seed!(8512)

    @testset "chatwithllm" begin

        c = ChatBot(p)

        d = chatwithllm(c; max_tokens=62)
        e = chatwithllm(c, " a secret path"; max_tokens=62)


        @test d isa String
        @test e isa String

        @test d == " Once upon a time, there was a little bird who loved singing. She flew everywhere in the forest, and sometimes even made up her own song for the other birds to sing along. One day, she met a little girl who was lost and couldn't find her way home. The little bird showed her"
        @test e == " a secret path that led the way to her home.\nThe little girl was grateful and thanked the bird. But as she started to leave, the little bird became selfish and scurried away. The little girl was sad and realized that she should have listened to the bird's advice and left the little"

        f = ChatBot(p)

        g = chatwithllm(f; max_tokens=300)

        @test g isa String

        @test g == " Anna liked to play with her toy house. She would pretend to be a mommy and a daddy, and she would dress up nails with her toys. She had a lot of fun with her toy house.\nOne day, Anna was playing with her toy house when she saw a small bird outside her window. The bird was chirping and cheerful, almost like a magic show. Anna wanted to see the bird closer, so she opened the window and climbed out.\nBut as soon as she went outside, the window started to crack. Anna strangely realized she was trapped inside. She tried to reach and push, but it was too fragile and hurt her hand. She screamed for help, but nobody saw her.\nSuddenly, the window broke and Anna was trapped inside. It was dark and scary. She hoped that everyone would hear her and come to help her. But no one did. Anna realized that pretending was not enough. She wished that she had a friend to help her. But nobody came. Anna was trapped in the room with no one to talk to. And sometimes, pretending is all that way."

        seed!(111134)
        
        f = ChatBot(p)

        g = chatwithllm(f; max_tokens=300)

        @test g isa String

        @test g == " Once upon a time, there was a little girl named Lily. She loved to play with her toys and explore her house. One day, Lily's mom asked her to clean up her toys, but Lily didn't want to. She wanted to go to explore some more.\nLily's mom said, \"If you can organize your toys, you will find your favorite doll.\" Lily sighed and started to put her toys away. She discovered her doll under her bed and then started to put her other toys away.\nAs Lily was organizing her toys, she found a journal that her mom had given her. She didn't know what was inside, but it looked important. Lily decided to keep it under her bed a little longer.\nLater that day, Lily's mom came into her room and asked her how she was organizing her toys. Lily proudly showed her mom the journal and told her all about her adventure. Her mom was happy to hear about it and suggested they make a special treasure out of Lys that they didn't need anymore.\nLily was excited and they went to the toy store and bought"
        
        i = ChatBot(p)

        h = chatwithllm(i, " and had the best day "; temperature=1.1f0, topp=0.5f0)

        @test h isa String

        @test h == " and had the best day denusted something dangerous! From then on, Amy remembered to never follow its journey with daddy again."
    end

end
