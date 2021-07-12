function assemble_face_robin_operator!(
    sysmatrix,
    basis,
    quad1,
    quad2,
    lambda,
    scaleareas,
    nodeids1,
    nodeids2,
)

    M11 = 0.25 * lambda * vec(mass_operator(basis, quad1, quad1, scaleareas))
    M12 = 0.25 * lambda * vec(mass_operator(basis, quad1, quad2, scaleareas))
    M21 = 0.25 * lambda * vec(mass_operator(basis, quad2, quad1, scaleareas))
    M22 = 0.25 * lambda * vec(mass_operator(basis, quad2, quad2, scaleareas))

    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        nodeids1,
        1,
        M11,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        nodeids2,
        1,
        M12,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids1,
        1,
        M21,
    )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids2,
        1,
        M22,
    )
end

function assemble_cell_robin_operator!(
    sysmatrix,
    basis,
    interfacequads,
    lambda,
    mesh,
    cellid,
)

    quad1 = interfacequads[+1, cellid]
    quad2 = interfacequads[-1, cellid]

    scaleareas = CutCellDG.interface_scale_areas(interfacequads, cellid)

    nodeids1 = CutCellDG.nodal_connectivity(mesh, +1, cellid)
    nodeids2 = CutCellDG.nodal_connectivity(mesh, -1, cellid)

    assemble_face_robin_operator!(
        sysmatrix,
        basis,
        quad1,
        quad2,
        lambda,
        scaleareas,
        nodeids1,
        nodeids2,
    )
end

function assemble_robin_operator!(
    sysmatrix,
    basis,
    interfacequads,
    lambda,
    mesh,
)
    ncells = CutCellDG.number_of_cells(mesh)

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)
        if cellsign == 0
            assemble_cell_robin_operator!(
                sysmatrix,
                basis,
                interfacequads,
                lambda,
                mesh,
                cellid,
            )
        end
    end
end
