module InteriorPenalty
using PolynomialBasis
using CutCellDG

include("IP_gradient_operator.jl")
include("IP_flux_operator.jl")
include("IP_penalty_operator.jl")
include("IP_robin_interface_operator.jl")
include("IP_robin_interface_source.jl")
include("IP_source_term.jl")
include("IP_assembly.jl")
include("IP_postprocess.jl")

end
