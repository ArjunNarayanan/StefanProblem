function gradient_operator(basis, quad, conductivity, jacobian, detjac)
    nf = number_of_basis_functions(basis)
    matrix = zeros(nf, nf)
    for (p, w) in quad
        G = CutCellDG.transform_gradient(gradient(basis, p), jacobian)
        matrix .+= conductivity * G * G' * detjac * w
    end
    return matrix
end

function assemble_cut_cell_gradient_operator!(
    sysmatrix,
    basis,
    cellquads,
    conductivity,
    mesh,
    cellsign,
    cellid,
)

    jac = CutCellDG.jacobian(mesh)
    detjac = CutCellDG.determinant_jacobian(mesh)
    quad = cellquads[cellsign, cellid]
    cellmatrix = vec(gradient_operator(basis, quad, conductivity, jac, detjac))
    nodeids = CutCellDG.nodal_connectivity(mesh, cellsign, cellid)
    CutCellDG.assemble_cell_matrix!(sysmatrix, nodeids, 1, cellmatrix)
end

function assemble_gradient_operator!(sysmatrix, basis, cellquads, k1, k2, mesh)
    ncells = CutCellDG.number_of_cells(mesh)
    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)
        if cellsign == +1 || cellsign == 0
            assemble_cut_cell_gradient_operator!(
                sysmatrix,
                basis,
                cellquads,
                k1,
                mesh,
                +1,
                cellid,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_cut_cell_gradient_operator!(
                sysmatrix,
                basis,
                cellquads,
                k2,
                mesh,
                -1,
                cellid,
            )
        end
    end
end
