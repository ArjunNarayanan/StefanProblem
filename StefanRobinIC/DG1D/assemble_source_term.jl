function linear_form(rhsfunc, basis, quad, cellmap, detjac)
    nf = number_of_basis_functions(basis)
    rhs = zeros(nf)
    for (p, w) in quad
        V = basis(p)
        gd = rhsfunc(cellmap(p))

        rhs .+= V * gd * detjac * w
    end
    return rhs
end

function assemble_cell_source!(
    systemrhs,
    rhsfunc,
    basis,
    quad,
    nodeids,
    cellmap,
)
    jacobian = CutCellDG.jacobian(cellmap)
    rhs = linear_form(rhsfunc, basis, quad, cellmap, jacobian)
    CutCellDG.assemble_cell_rhs!(systemrhs, nodeids, 1, rhs)
end

function assemble_two_phase_source!(
    systemrhs,
    rhsfunc1,
    rhsfunc2,
    basis,
    quad,
    mesh,
)
    ncells = number_of_cells(mesh)
    for cellid = 1:ncells
        label = element_label(mesh, cellid)
        nodeids = nodal_connectivity(mesh, cellid)
        cellmap = cell_map(mesh, cellid)
        if label == 1
            assemble_cell_source!(
                systemrhs,
                rhsfunc1,
                basis,
                quad,
                nodeids,
                cellmap,
            )
        elseif label == 2
            assemble_cell_source!(
                systemrhs,
                rhsfunc2,
                basis,
                quad,
                nodeids,
                cellmap,
            )
        else
            error("Expected label = {1,2}, got label = $label")
        end
    end
end
