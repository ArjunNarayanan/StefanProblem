function robin_interface_linear_form(basis, quad, scaleareas)
    @assert length(quad) == length(scaleareas)
    nf = number_of_basis_functions(basis)
    rhs = zeros(nf)
    for (idx, (p, w)) in enumerate(quad)
        V = basis(p)
        rhs .+= V * scaleareas[idx] * w
    end
    return rhs
end

function assemble_face_robin_source!(
    sysrhs,
    basis,
    quad,
    lambda,
    Tm,
    scaleareas,
    nodeids,
)

    rhs =
        0.5 * lambda * Tm * robin_interface_linear_form(basis, quad, scaleareas)
    CutCellDG.assemble_cell_rhs!(sysrhs, nodeids, 1, rhs)
end

function assemble_cell_robin_source!(
    sysrhs,
    basis,
    interfacequads,
    lambda,
    Tm,
    mesh,
    cellid,
)

    quad1 = interfacequads[+1, cellid]
    quad2 = interfacequads[-1, cellid]

    scaleareas = CutCellDG.interface_scale_areas(interfacequads, cellid)

    nodeids1 = CutCellDG.nodal_connectivity(mesh, +1, cellid)
    nodeids2 = CutCellDG.nodal_connectivity(mesh, -1, cellid)

    assemble_face_robin_source!(
        sysrhs,
        basis,
        quad1,
        lambda,
        Tm,
        scaleareas,
        nodeids1,
    )
    assemble_face_robin_source!(
        sysrhs,
        basis,
        quad2,
        lambda,
        Tm,
        scaleareas,
        nodeids2,
    )
end

function assemble_robin_source!(sysrhs, basis, interfacequads, lambda, Tm, mesh)
    ncells = CutCellDG.number_of_cells(mesh)

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)
        if cellsign == 0
            assemble_cell_robin_source!(
                sysrhs,
                basis,
                interfacequads,
                lambda,
                Tm,
                mesh,
                cellid,
            )
        end
    end
end
