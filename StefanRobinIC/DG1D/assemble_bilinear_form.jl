function gradient_operator(basis, quad, conductivity, jacobian)
    nf = number_of_basis_functions(basis)
    matrix = zeros(nf, nf)
    for (p, w) in quad
        G = gradient(basis, p) ./ jacobian
        matrix .+= conductivity * G * G' * jacobian * w
    end
    return matrix
end

function assemble_cell_gradient_operator!(
    sysmatrix,
    basis,
    quad,
    conductivity,
    mesh,
    cellid,
)
    jacobian = CutCellDG.jacobian(cell_map(mesh, cellid))
    cellmatrix = vec(gradient_operator(basis, quad, conductivity, jacobian))
    nodeids = nodal_connectivity(mesh, cellid)
    CutCellDG.assemble_cell_matrix!(sysmatrix, nodeids, 1, cellmatrix)
end

function assemble_gradient_operator!(sysmatrix, basis, quad, k1, k2, mesh)
    nelmts = number_of_elements(mesh)
    for cellid = 1:nelmts
        if element_label(mesh, cellid) == 1
            assemble_cell_gradient_operator!(
                sysmatrix,
                basis,
                quad,
                k1,
                mesh,
                cellid,
            )
        elseif element_label(mesh, cellid) == 2
            assemble_cell_gradient_operator!(
                sysmatrix,
                basis,
                quad,
                k2,
                mesh,
                cellid,
            )
        end
    end
end
