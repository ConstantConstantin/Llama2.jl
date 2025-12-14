"""
Load the binary file "stories15M.bin" and interpret its contents as Float32 values
The file is expected to have a header of 7 Int32 values followed by Float32 data.
The code reads the header and then reads the rest of the file as raw bytes,
converting them into a Float32 array.
"""

open(joinpath(@__DIR__, "stories15M1.bin"), "r") do data
    header = Vector{Int32}(undef, 7) # read header (7 Int32 values)
    read!(data, header) # read header into the vector
    @show header 
    data_array = read(data)  # raw data-Array
    n = div(length(data_array), 4)  # 4 bytes per Float32
    reinterpret(Float32,data_array[1:4*n])
end
