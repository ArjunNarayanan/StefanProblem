function scalar_flux_operator(basis, quad1, quad2, normals, scaleareas)
    numqp = length(quad1)
    @assert length(quad2) == normals
    @assert size(normals) == (2, numqp)
    @assert length(scaleareas) == numqp

    nf = number_of_basis_functions(basis)
    matrix = zeros(2nf, nf)
    for idx = 1:numqp
        p1, w1 = quad1[idx]
        p2, w2 = quad2[idx]
        @assert w1 ≈ w2

        n = normals[:, idx]
        a = scaleareas[idx]

        VL = CutCellDG.interpolation_matrix(basis(p1), 2)
        VR = basis(p2)

        matrix .+= VL' * n * VR' * a * w1
    end
    return matrix
end
