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

function assemble_face_vector_flux_operator!(
    sysmatrix,
    basis,
    quad1,
    quad2,
    normals,
    conductivity1,
    conductivity2,
    alpha,
    beta,
    scaleareas,
    nodeids1,
    nodeids2,
)

    posnormals = normals
    negnormals = -normals

    M11 =
        -0.5 *
        conductivity1 *
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
        conductivity2 *
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

    nnp = vec(mapslices(sum, posnormals .* posnormals, dims = 1))
    nnm = vec(mapslices(sum, negnormals .* posnormals, dims = 1))

    N11 = alpha * vec(mass_operator(basis, quad1, quad1, 1, nnp .* scaleareas))
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        [1],
        nodeids1,
        [1],
        3,
        N11,
    )
    N12 = alpha * vec(mass_operator(basis, quad1, quad2, 1, nnm .* scaleareas))
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        [1],
        nodeids2,
        [1],
        3,
        N12,
    )

    bn = posnormals' * beta
    L11 =
        conductivity1 * vec(
            vector_flux_operator(
                basis,
                quad1,
                quad1,
                posnormals,
                bn .* scaleareas,
            ),
        )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        [1],
        nodeids1,
        [2, 3],
        3,
        L11,
    )
    L12 =
        conductivity2 * vec(
            vector_flux_operator(
                basis,
                quad1,
                quad2,
                negnormals,
                bn .* scaleareas,
            ),
        )
    CutCellDG.assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        [1],
        nodeids2,
        [2, 3],
        3,
        L12,
    )
end


################################################################################
function assemble_face_interelement_vector_flux_operator!(
    sysmatrix,
    basis,
    quad1,
    quad2,
    normal,
    conductivity,
    alpha,
    beta,
    facedetjac,
    nodeids1,
    nodeids2,
)

    numqp = length(quad1)
    normals = repeat(normal, inner = (1, numqp))
    scaleareas = repeat([facedetjac], numqp)

    assemble_face_vector_flux_operator!(
        sysmatrix,
        basis,
        quad1,
        quad2,
        normals,
        conductivity,
        conductivity,
        alpha,
        beta,
        scaleareas,
        nodeids1,
        nodeids2,
    )
end

function assemble_cut_cell_interelement_vector_flux_operator!(
    sysmatrix,
    basis,
    facequads,
    normals,
    conductivity,
    alpha,
    beta,
    mesh,
    cellsign,
    cellid,
    faceids,
    nbrfaceids,
    facedetjac,
)

    nodeids1 = CutCellDG.nodal_connectivity(mesh, cellsign, cellid)
    for faceid in faceids
        quad1 = facequads[cellsign, faceid, cellid]
        nbrcellid = CutCellDG.cell_connectivity(mesh, faceid, cellid)

        if nbrcellid != 0 &&
           CutCellDG.solution_cell_id(mesh, cellsign, nbrcellid) !=
           CutCellDG.solution_cell_id(mesh, cellsign, cellid)

            nbrfaceid = nbrfaceids[faceid]
            nbrcellsign = CutCellDG.cell_sign(mesh, nbrcellid)

            if nbrcellsign == cellsign || nbrcellsign == 0
                quad2 = facequads[cellsign, nbrfaceid, nbrcellid]
                nodeids2 =
                    CutCellDG.nodal_connectivity(mesh, cellsign, nbrcellid)

                assemble_face_interelement_vector_flux_operator!(
                    sysmatrix,
                    basis,
                    quad1,
                    quad2,
                    normals[faceid],
                    conductivity,
                    alpha,
                    beta,
                    facedetjac[faceid],
                    nodeids1,
                    nodeids2,
                )
            end
        end
    end
end

function assemble_interelement_vector_flux_operator!(
    sysmatrix,
    basis,
    facequads,
    conductivity1,
    conductivity2,
    alpha,
    beta,
    mesh,
)

    ncells = CutCellDG.number_of_cells(mesh)
    facedetjac = CutCellDG.face_determinant_jacobian(mesh)
    nfaces = CutCellDG.number_of_faces_per_cell(facequads)
    faceids = 1:nfaces
    nbrfaceids = [CutCellDG.opposite_face(faceid) for faceid in faceids]
    normals = CutCellDG.reference_face_normals()

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)

        if cellsign == +1 || cellsign == 0
            assemble_cut_cell_interelement_vector_flux_operator!(
                sysmatrix,
                basis,
                facequads,
                normals,
                conductivity1,
                alpha,
                beta,
                mesh,
                +1,
                cellid,
                faceids,
                nbrfaceids,
                facedetjac,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_cut_cell_interelement_vector_flux_operator!(
                sysmatrix,
                basis,
                facequads,
                normals,
                conductivity2,
                alpha,
                beta,
                mesh,
                -1,
                cellid,
                faceids,
                nbrfaceids,
                facedetjac,
            )
        end
    end
end
################################################################################


################################################################################
function assemble_cell_interface_vector_flux_operator!(
    sysmatrix,
    basis,
    interfacequads,
    k1,
    k2,
    alpha,
    beta,
    mesh,
    cellid,
)

    quad1 = interfacequads[+1, cellid]
    quad2 = interfacequads[-1, cellid]

    negnormals = CutCellDG.interface_normals(interfacequads, cellid)
    posnormals = -1.0 * negnormals
    scaleareas = CutCellDG.interface_scale_areas(interfacequads, cellid)

    nodeids1 = CutCellDG.nodal_connectivity(mesh, +1, cellid)
    nodeids2 = CutCellDG.nodal_connectivity(mesh, -1, cellid)

    assemble_face_vector_flux_operator!(
        sysmatrix,
        basis,
        quad1,
        quad2,
        posnormals,
        k1,
        k2,
        alpha,
        beta,
        scaleareas,
        nodeids1,
        nodeids2,
    )
    assemble_face_vector_flux_operator!(
        sysmatrix,
        basis,
        quad2,
        quad1,
        negnormals,
        k2,
        k1,
        alpha,
        beta,
        scaleareas,
        nodeids2,
        nodeids1,
    )
end

function assemble_interface_vector_flux_operator!(
    sysmatrix,
    basis,
    interfacequads,
    k1,
    k2,
    alpha,
    beta,
    mesh,
)

    ncells = CutCellDG.number_of_cells(mesh)
    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)
        if cellsign == 0
            assemble_cell_interface_vector_flux_operator!(
                sysmatrix,
                basis,
                interfacequads,
                k1,
                k2,
                alpha,
                beta,
                mesh,
                cellid,
            )
        end
    end
end
################################################################################


################################################################################
function assemble_boundary_face_vector_flux_operator!(
    sysmatrix,
    basis,
    quad,
    normal,
    conductivity,
    alpha,
    facedetjac,
    nodeids,
)

    numqp = length(quad)

    if numqp > 0
        posnormals = repeat(normal, inner = (1, numqp))
        negnormals = -posnormals
        scaleareas = repeat([facedetjac], numqp)

        M11 =
            -conductivity *
            vec(vector_flux_operator(basis, quad, quad, posnormals, scaleareas))
        CutCellDG.assemble_couple_cell_matrix!(
            sysmatrix,
            nodeids,
            [1],
            nodeids,
            [2, 3],
            3,
            M11,
        )

        nnp = vec(mapslices(sum, posnormals .* posnormals, dims = 1))
        nnm = vec(mapslices(sum, negnormals .* posnormals, dims = 1))

        N11 = alpha * vec(mass_operator(basis, quad, quad, 1, nnp .* scaleareas))
        CutCellDG.assemble_couple_cell_matrix!(
            sysmatrix,
            nodeids,
            [1],
            nodeids,
            [1],
            3,
            N11,
        )
    end
end

function assemble_boundary_cell_vector_flux_operator!(
    sysmatrix,
    basis,
    facequads,
    normals,
    conductivity,
    alpha,
    mesh,
    cellsign,
    cellid,
    faceids,
    facedetjac,
)
    nodeids = CutCellDG.nodal_connectivity(mesh, cellsign, cellid)
    for faceid in faceids
        quad = facequads[cellsign, faceid, cellid]
        nbrcellid = CutCellDG.cell_connectivity(mesh, faceid, cellid)

        if nbrcellid == 0 # this is a boundary face
            assemble_boundary_face_vector_flux_operator!(
                sysmatrix,
                basis,
                quad,
                normals[faceid],
                conductivity,
                alpha,
                facedetjac[faceid],
                nodeids,
            )
        end
    end
end

function assemble_boundary_vector_flux_operator!(
    sysmatrix,
    basis,
    facequads,
    k1,
    k2,
    alpha,
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
            assemble_boundary_cell_vector_flux_operator!(
                sysmatrix,
                basis,
                facequads,
                normals,
                k1,
                alpha,
                mesh,
                +1,
                cellid,
                faceids,
                facedetjac,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_boundary_cell_vector_flux_operator!(
                sysmatrix,
                basis,
                facequads,
                normals,
                k2,
                alpha,
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
