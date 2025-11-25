using Llama2
using Documenter

DocMeta.setdocmeta!(Llama2, :DocTestSetup, :(using Llama2); recursive=true)

makedocs(;
    modules=[Llama2],
    authors="Constantin-Paul Hertel <paul.hertel@aol.de>",
    sitename="Llama2.jl",
    format=Documenter.HTML(;
        canonical="https://ConstantConstantin.github.io/Llama2.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ConstantConstantin/Llama2.jl",
    devbranch="main",
)
