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
    CutCellDG.assemble_cell_rhs!(systemrhs, nodeids, [1], 3, rhs)
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
function boundary_face_scalar_flux(
    rhsfunc,
    basis,
    quad,
    normals,
    cellmap,
    scaleareas,
)

    numqp = length(quad)
    @assert size(normals) == (2,numqp)
    @assert length(scaleareas) == numqp

    nf = number_of_basis_functions(basis)
    rhs = zeros(2nf)
    for (idx,(p, w)) in enumerate(quad)
        V2 = CutCellDG.interpolation_matrix(basis(p), 2)
        gd = rhsfunc(cellmap(p))

        n = normals[:,idx]
        a = scaleareas[idx]

        rhs .+= V2' * n * gd * a * w
    end
    return rhs
end

function assemble_boundary_face_scalar_flux_source!(
    systemrhs,
    rhsfunc,
    basis,
    quad,
    normal,
    cellmap,
    facedetjac,
    nodeids,
)

    numqp = length(quad)
    normals = repeat(normal,inner=(1,numqp))
    scaleareas = repeat([facedetjac],numqp)

    rhs = boundary_face_scalar_flux(
        rhsfunc,
        basis,
        quad,
        normals,
        cellmap,
        scaleareas,
    )
    CutCellDG.assemble_cell_rhs!(systemrhs, nodeids, [2, 3], 3, rhs)
end

function assemble_boundary_cell_scalar_flux_source!(
    systemrhs,
    rhsfunc,
    basis,
    facequads,
    normals,
    mesh,
    cellsign,
    cellid,
    faceids,
    facedetjac,
)

    nodeids = CutCellDG.nodal_connectivity(mesh, cellsign, cellid)
    cellmap = CutCellDG.cell_map(mesh, cellsign, cellid)

    for faceid in faceids
        quad = facequads[cellsign, faceid, cellid]
        nbrcellid = CutCellDG.cell_connectivity(mesh, faceid, cellid)

        if nbrcellid == 0
            assemble_boundary_face_scalar_flux_source!(
                systemrhs,
                rhsfunc,
                basis,
                quad,
                normals[faceid],
                cellmap,
                facedetjac[faceid],
                nodeids,
            )
        end
    end
end

function assemble_boundary_scalar_flux_source!(
    systemrhs,
    rhsfunc,
    basis,
    facequads,
    mesh,
)
    ncells = CutCellDG.number_of_cells(mesh)
    facedetjac = CutCellDG.face_determinant_jacobian(mesh)
    normals = CutCellDG.reference_face_normals()
    nfaces = CutCellDG.number_of_faces_per_cell(facequads)
    faceids = 1:nfaces

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)

        if cellsign == +1 || cellsign == 0
            assemble_boundary_cell_scalar_flux_source!(
                systemrhs,
                rhsfunc,
                basis,
                facequads,
                normals,
                mesh,
                +1,
                cellid,
                faceids,
                facedetjac,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_boundary_cell_scalar_flux_source!(
                systemrhs,
                rhsfunc,
                basis,
                facequads,
                normals,
                mesh,
                -1,
                cellid,
                faceids,
                facedetjac,
            )
        end
    end
end
################################################################################


################################################################################
function assemble_boundary_face_vector_flux_source!(
    systemrhs,
    rhsfunc,
    basis,
    quad,
    normal,
    negboundarypenalty,
    posboundarypenalty,
    V0,
    cellmap,
    facedetjac,
    nodeids,
)

    V0nk = V0'*normal
    alpha = V0nk < 0 ? negboundarypenalty : posboundarypenalty

    nnp = (-normal)' * normal
    rhs = -alpha * linear_form(rhsfunc, basis, quad, cellmap, nnp * facedetjac)
    CutCellDG.assemble_cell_rhs!(systemrhs, nodeids, [1], 3, rhs)
end

function assemble_boundary_cell_vector_flux_source!(
    systemrhs,
    rhsfunc,
    basis,
    facequads,
    normals,
    negboundarypenalty,
    posboundarypenalty,
    V0,
    mesh,
    cellsign,
    cellid,
    faceids,
    facedetjac,
)

    nodeids = CutCellDG.nodal_connectivity(mesh, cellsign, cellid)
    cellmap = CutCellDG.cell_map(mesh, cellsign, cellid)
    for faceid in faceids
        quad = facequads[cellsign, faceid, cellid]
        nbrcellid = CutCellDG.cell_connectivity(mesh, faceid, cellid)

        if nbrcellid == 0
            assemble_boundary_face_vector_flux_source!(
                systemrhs,
                rhsfunc,
                basis,
                quad,
                normals[faceid],
                negboundarypenalty,
                posboundarypenalty,
                V0,
                cellmap,
                facedetjac[faceid],
                nodeids,
            )
        end
    end
end

function assemble_boundary_vector_flux_source!(
    systemrhs,
    rhsfunc,
    basis,
    facequads,
    negboundarypenalty,
    posboundarypenalty,
    V0,
    mesh,
)

    ncells = CutCellDG.number_of_cells(mesh)
    facedetjac = CutCellDG.face_determinant_jacobian(mesh)
    nfaces = CutCellDG.number_of_faces_per_cell(facequads)
    faceids = 1:nfaces
    normals = CutCellDG.reference_face_normals()

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)

        if cellsign == +1 || cellsign == 0
            assemble_boundary_cell_vector_flux_source!(
                systemrhs,
                rhsfunc,
                basis,
                facequads,
                normals,
                negboundarypenalty,
                posboundarypenalty,
                V0,
                mesh,
                +1,
                cellid,
                faceids,
                facedetjac,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_boundary_cell_vector_flux_source!(
                systemrhs,
                rhsfunc,
                basis,
                facequads,
                normals,
                negboundarypenalty,
                posboundarypenalty,
                V0,
                mesh,
                -1,
                cellid,
                faceids,
                facedetjac,
            )
        end
    end
end
################################################################################
