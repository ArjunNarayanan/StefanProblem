function assemble_cut_cell_mass_operator!(
    sysmatrix,
    basis,
    cellquads,
    mesh,
    cellsign,
    cellid,
)

    detjac = CutCellDG.determinant_jacobian(mesh)
    quad = cellquads[cellsign, cellid]

    cellmatrix = vec(CutCellDG.mass_matrix(basis, quad, 2, detjac))

    nodeids = CutCellDG.nodal_connectivity(mesh, cellsign, cellid)
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids,
        [2, 3],
        nodeids,
        [2, 3],
        3,
        cellmatrix,
    )
end

function assemble_mass_operator!(sysmatrix, basis, cellquads, mesh)
    ncells = CutCellDG.number_of_cells(mesh)
    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)
        if cellsign == +1 || cellsign == 0
            assemble_cut_cell_mass_operator!(
                sysmatrix,
                basis,
                cellquads,
                mesh,
                +1,
                cellid,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_cut_cell_mass_operator!(
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
