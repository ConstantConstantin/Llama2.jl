```@meta
CurrentModule = Llama2
```

# Llama2.jl

## What is Llama2?

Llama 2 is a family of pre-trained large language models (LLMs) developed by Meta AI. It includes a range of model sizes, from 7 to 70 billion parameters. Llama 2 models use a classical Transformer architecture with slight variations, most notably the use of RMSNorm instead of standard layer normalization, and the SiLU (Sigmoid Linear Unit) activation function rather than ReLU.

It was released in June 2023. More information can be found at: https://www.llama.com/llama2/

## What is Llama2.jl?

[Llama2.jl](https://github.com/ConstantConstantin/Llama2.jl) can perform inference on any given Llama 2 model from within `julia`. Inference refers to the use of a trained model for tasks such as chatting. It involves mapping user inputs to a predefined vocabulary through encoding and decoding functions, using the model’s weights and architecture to generate outputs, and sampling those outputs to construct a prompt in a desired manner. It includes a continuous chat function that keeps past inputs in a struct, making ongoing conversations possible.

Its core functionality is written entirely in Julia, with very minimal reliance on other Julia packages.

For this reason, you must provide your own model checkpoint. This project follows the procedure outlined in the `run.c` file from [llama2.c](https://github.com/karpathy/llama2.c).

## Getting started

Start julia, activate a desired environment and add the package:

```julia
(@v1.11) pkg> activate .

(myLlama2) pkg> add https://github.com/ConstantConstantin/Llama2.jl
```

In every subsequent session it can be loaded via:

```julia
julia> using Llama2
```

## Example Usage
For using Interference, a 
There are two main functions for chatting in this package: `talktollm` and `chatwithllm`. `talktollm` is intended for out-of-the-box use and demonstrates the core functionality of the package. It requires only a model as input, with optional arguments such as text input, temperature, or maximum token length. Using `talktollm`, you can generate only a single prompt.

```julia
julia> print(talktollm("/PATH/TO/YOUR/MODEL.bin", "In a small village "))
In a small village house, there was a man named Tom. Tom was kind and would always shine his in front of the town. People from the village would come to look at Tom and feel happy.
One day, a little girl named Lily came to Tom. She did not have a passport. Tom saw Lily and said, "Why don't you have a passport, Lily? Hop in and pass me a little in our country!" Lily smiled and said, "Yes, I feel comfortable when I am in my own nation!"
Lily put on her sunglasses and they became good friends. The town was filled with happy puppies who shared their sunglasses with everyone. The people in the town knew that being kind and working together made everything better.
```

For a continoud chat, initialize a Struct, containing the Runstate of you Conversation. This `ChatBot` Struct is always using only one Model, to have multiple Models in your Session, generate multiple Structs.

```julia
julia> c = ChatBot("/PATH/TO/YOUR/MODEL.bin");
```

This then gets Updated with a Chatfunction `chatwithllm`, taking a Textinput. 

```julia
julia> print(chatwithllm(c))
 Once upon a time, there was an old house with an ancient sign inside. The sign was very big and had many words on it. One day, a little girl went to visit the old house. She wanted to see what was inside.
The old house said, "Hello? Can I come in?"

julia> print(chatwithllm(c, "\nThe little girl said:"))

The little girl said: "Yes please! Can I come in too?"
The old house thought for moments before it said, "Yes. This light is available for you 30 cent a nightmare."
The little girl was very excited. She said thank you and then, followed her favorite sign
```

The provided example model is a storyteller capable of generating a random story or continuing a story initiated by the user. More Models can be found 

## API

```@index
```
