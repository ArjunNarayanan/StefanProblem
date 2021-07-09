################################################################################
function assemble_cell_source!(
    systemrhs,
    rhsfunc,
    basis,
    cellquads,
    mesh,
    cellsign,
    cellid,
)
    detjac = CutCellDG.determinant_jacobian(mesh)
    cellmap = CutCellDG.cell_map(mesh, cellsign, cellid)
    quad = cellquads[cellsign, cellid]
    rhs = linear_form(rhsfunc, basis, quad, cellmap, detjac)
    nodeids = CutCellDG.nodal_connectivity(mesh, cellsign, cellid)
    CutCellDG.assemble_cell_rhs!(systemrhs, nodeids, 1, rhs)
end

function assemble_source!(systemrhs, rhsfunc, basis, cellquads, mesh)
    ncells = CutCellDG.number_of_cells(mesh)

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)

        if cellsign == +1 || cellsign == 0
            assemble_cell_source!(
                systemrhs,
                rhsfunc,
                basis,
                cellquads,
                mesh,
                +1,
                cellid,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_cell_source!(
                systemrhs,
                rhsfunc,
                basis,
                cellquads,
                mesh,
                -1,
                cellid,
            )
        end
    end
end
################################################################################


################################################################################
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

function flux_linear_form(
    rhsfunc,
    basis,
    quad,
    normal,
    conductivity,
    cellmap,
    jacobian,
    facedetjac,
)
    nf = number_of_basis_functions(basis)
    rhs = zeros(nf)

    for (idx, (p, w)) in enumerate(quad)
        G = CutCellDG.transform_gradient(gradient(basis, p), jacobian)
        R = rhsfunc(cellmap(p))

        rhs .+= conductivity * G * normal * R * facedetjac * w
    end
    return rhs
end

function assemble_boundary_face_source!(
    systemrhs,
    rhsfunc,
    basis,
    quad,
    normal,
    conductivity,
    penalty,
    cellmap,
    jacobian,
    facedetjac,
    nodeids,
)

    R1 = penalty * linear_form(rhsfunc, basis, quad, cellmap, facedetjac)
    R2 = flux_linear_form(
        rhsfunc,
        basis,
        quad,
        normal,
        conductivity,
        cellmap,
        jacobian,
        facedetjac,
    )
    rhs = R1 - R2

    CutCellDG.assemble_cell_rhs!(systemrhs, nodeids, 1, rhs)
end

function assemble_boundary_cell_source!(
    systemrhs,
    rhsfunc,
    basis,
    facequads,
    normals,
    conductivity,
    penalty,
    mesh,
    cellsign,
    cellid,
    faceids,
    jacobian,
    facedetjac,
)

    nodeids = CutCellDG.nodal_connectivity(mesh, cellsign, cellid)
    cellmap = CutCellDG.cell_map(mesh, cellsign, cellid)
    for faceid in faceids
        quad = facequads[cellsign, faceid, cellid]
        nbrcellid = CutCellDG.cell_connectivity(mesh, faceid, cellid)

        if nbrcellid == 0
            assemble_boundary_face_source!(
                systemrhs,
                rhsfunc,
                basis,
                quad,
                normals[faceid],
                conductivity,
                penalty,
                cellmap,
                jacobian,
                facedetjac[faceid],
                nodeids,
            )
        end
    end
end

function assemble_boundary_source!(
    systemrhs,
    rhsfunc,
    basis,
    facequads,
    k1,
    k2,
    penalty,
    mesh,
)

    ncells = CutCellDG.number_of_cells(mesh)
    jacobian = CutCellDG.jacobian(mesh)
    facedetjac = CutCellDG.face_determinant_jacobian(mesh)
    nfaces = CutCellDG.number_of_faces_per_cell(facequads)
    faceids = 1:nfaces
    normals = CutCellDG.reference_face_normals()

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)

        if cellsign == +1 || cellsign == 0
            assemble_boundary_cell_source!(
                systemrhs,
                rhsfunc,
                basis,
                facequads,
                normals,
                k1,
                penalty,
                mesh,
                +1,
                cellid,
                faceids,
                jacobian,
                facedetjac,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_boundary_cell_source!(
                systemrhs,
                rhsfunc,
                basis,
                facequads,
                normals,
                k2,
                penalty,
                mesh,
                -1,
                cellid,
                faceids,
                jacobian,
                facedetjac,
            )
        end
    end
end
################################################################################
