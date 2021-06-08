function divergence_operator(basis, quad, jacobian, detjac)
    nf = number_of_basis_functions(basis)
    matrix = zeros(2nf, nf)
    for (p, w) in quad
        G = CutCellDG.transform_gradient(gradient(basis, p), jacobian)
        DivG = vec(G')
        V = basis(p)
        matrix .+= DivG * V' * detjac * w
    end
    return matrix
end

function assemble_cut_cell_divergence_operator!(
    sysmatrix,
    basis,
    cellquads,
    mesh,
    cellsign,
    cellid,
)

    jac = CutCellDG.jacobian(mesh)
    detjac = CutCellDG.determinant_jacobian(mesh)
    quad = cellquads[cellsign, cellid]

    cellmatrix = vec(divergence_operator(basis, quad, jac, detjac))

    nodeids = CutCellDG.nodal_connectivity(mesh, cellsign, cellid)
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids,
        [2, 3],
        nodeids,
        [1],
        3,
        cellmatrix,
    )
end

function assemble_divergence_operator!(sysmatrix, basis, cellquads, mesh)
    ncells = CutCellDG.number_of_cells(mesh)
    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)
        if cellsign == +1 || cellsign == 0
            assemble_cut_cell_divergence_operator!(
                sysmatrix,
                basis,
                cellquads,
                mesh,
                +1,
                cellid,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_cut_cell_divergence_operator!(
                sysmatrix,
                basis,
                cellquads,
                mesh,
                -1,
                cellid,
            )
        end
    end
end
