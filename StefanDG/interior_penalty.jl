module InteriorPenalty
using PolynomialBasis
using CutCellDG

include("IP_gradient_operator.jl")
include("IP_flux_operator.jl")
include("IP_penalty_operator.jl")
include("source_term.jl")
include("IP_assembly.jl")

end
