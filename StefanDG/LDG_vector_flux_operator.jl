function vector_flux_operator(basis, quad1, quad2, normals, scaleareas)

    numqp = length(quad1)
    @assert length(quad2) == numqp
    @assert size(normals) == (2, numqp)
    @assert length(scaleareas) == numqp

    nf = number_of_basis_functions(basis)
    matrix = zeros(nf, 2nf)
    for idx = 1:numqp
        p1, w1 = quad1[idx]
        p2, w2 = quad2[idx]
        @assert w1 ≈ w2

        n = normals[:, idx]
        a = scaleareas[idx]

        V = basis(p1)
        V2 = CutCellDG.interpolation_matrix(basis(p2), 2)

        matrix .+= V * n' * V2 * a * w1
    end
    return matrix
end

function mass_operator(basis, quad1, quad2, ndofs, scaleareas)
    numqp = length(quad1)
    @assert length(quad2) == numqp
    @assert length(scaleareas) == numqp

    nf = number_of_basis_functions(basis)
    matrix = zeros(nf, nf)
    for idx = 1:length(quad1)
        p1, w1 = quad1[idx]
        p2, w2 = quad2[idx]

        V1 = CutCellDG.interpolation_matrix(basis(p1), ndofs)
        V2 = CutCellDG.interpolation_matrix(basis(p2), ndofs)

        matrix .+= V1' * V2 * scaleareas[idx] * w1
    end
    return matrix
end

function assemble_vector_flux_operator!(
    sysmatrix,
    basis,
    quad1,
    quad2,
    normals,
    conductivity,
    alpha,
    beta,
    scaleareas,
    nodeids1,
    nodeids2,
)

    M11 =
        -0.5 *
        conductivity *
        vec(vector_flux_operator(basis, quad1, quad1, normals, scaleareas))
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        [1],
        nodeids1,
        [2, 3],
        3,
        M11,
    )
    M12 =
        -0.5 *
        conductivity *
        vector_flux_operator(basis, quad1, quad2, normals, scaleareas)
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        [1],
        nodeids2,
        [2, 3],
        3,
        M12,
    )

    nnp = vec(mapslices(sum, normals .* normals, dims = 1))
    nnm = vec(mapslices(sum, normals .* normals, dims = 1))

    N11 =
        alpha *
        conductivity *
        vec(mass_operator(basis, quad1, quad1, 1, nnp .* scaleareas))
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        [1],
        nodeids1,
        [1],
        3,
        N11,
    )
    N12 =
        alpha *
        conductivity *
        vec(mass_operator(basis, quad1, quad2, 1, nnm .* scaleareas))
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        [1],
        nodeids2,
        [1],
        3,
        N12,
    )

    
end
