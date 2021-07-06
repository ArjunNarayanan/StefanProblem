import Base.==, Base.≈

function allapprox(v1, v2)
    return all(v1 .≈ v2)
end

function allapprox(v1, v2, atol)
    err = maximum(abs.(v1 - v2))
    # if 10err > atol
    #     @warn "Approximation is very tight!"
    # end
    return length(v1) == length(v2) &&
           all([isapprox(v1[i], v2[i], atol = atol) for i = 1:length(v1)])
end

function allequal(v1, v2)
    return all(v1 .== v2)
end

function required_quadrature_order(polyorder)
    ceil(Int, 0.5 * (2polyorder + 1))
end

function add_cell_error_squared!(
    err,
    interpolater,
    exactsolution,
    cellmap,
    quad,
)
    detjac = CutCellDG.determinant_jacobian(cellmap)
    for (p, w) in quad
        numsol = interpolater(p)
        exsol = exactsolution(cellmap(p))
        err .+= (numsol - exsol) .^ 2 * detjac * w
    end
end

function mesh_L2_error(nodalsolutions, exactsolution, basis, cellquads, mesh)
    ndofs = size(nodalsolutions)[1]
    err = zeros(ndofs)
    interpolater = InterpolatingPolynomial(ndofs, basis)
    ncells = CutCellDG.number_of_cells(mesh)
    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)
        if cellsign == +1 || cellsign == 0
            cellmap = CutCellDG.cell_map(mesh, +1, cellid)
            nodeids = CutCellDG.nodal_connectivity(mesh, +1, cellid)
            elementsolution = nodalsolutions[:, nodeids]
            update!(interpolater, elementsolution)
            quad = cellquads[+1, cellid]
            add_cell_error_squared!(
                err,
                interpolater,
                exactsolution,
                cellmap,
                quad,
            )
        end
        if cellsign == -1 || cellsign == 0
            cellmap = CutCellDG.cell_map(mesh, -1, cellid)
            nodeids = CutCellDG.nodal_connectivity(mesh, -1, cellid)
            elementsolution = nodalsolutions[:, nodeids]
            update!(interpolater, elementsolution)
            quad = cellquads[-1, cellid]
            add_cell_error_squared!(
                err,
                interpolater,
                exactsolution,
                cellmap,
                quad,
            )
        end
    end
    return sqrt.(err)
end

function cellwise_L2_error(
    nodalsolutions,
    exactsolution,
    basis,
    cellquads,
    mesh,
)

    ndofs = size(nodalsolutions)[1]
    interpolater = InterpolatingPolynomial(ndofs, basis)
    ncells = CutCellDG.number_of_cells(mesh)
    err = zeros(ndofs, 2, ncells)

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)
        if cellsign == +1 || cellsign == 0
            cellmap = CutCellDG.cell_map(mesh, +1, cellid)
            nodeids = CutCellDG.nodal_connectivity(mesh, +1, cellid)
            elementsolution = nodalsolutions[:, nodeids]
            update!(interpolater, elementsolution)
            quad = cellquads[+1, cellid]
            cellerr = zeros(ndofs)
            add_cell_error_squared!(
                cellerr,
                interpolater,
                exactsolution,
                cellmap,
                quad,
            )
            err[:, 1, cellid] = cellerr
        end
        if cellsign == -1 || cellsign == 0
            cellmap = CutCellDG.cell_map(mesh, -1, cellid)
            nodeids = CutCellDG.nodal_connectivity(mesh, -1, cellid)
            elementsolution = nodalsolutions[:, nodeids]
            update!(interpolater, elementsolution)
            quad = cellquads[-1, cellid]
            cellerr = zeros(ndofs)
            add_cell_error_squared!(
                cellerr,
                interpolater,
                exactsolution,
                cellmap,
                quad,
            )
            err[:, 2, cellid] = cellerr
        end
    end
    return sqrt.(err)
end

function average(v)
    return sum(v) / length(v)
end

function convergence_rate(dx, err)
    return diff(log.(err)) ./ diff(log.(dx))
end

function add_cell_norm_squared!(vals, func, cellmap, quad)
    detjac = CutCellDG.determinant_jacobian(cellmap)
    for (p, w) in quad
        v = func(cellmap(p))
        vals .+= v .^ 2 * detjac * w
    end
end

function integral_norm_on_mesh(func, cellquads, mesh, ndofs)
    vals = zeros(ndofs)
    ncells = CutCellDG.number_of_cells(mesh)
    for cellid = 1:ncells
        s = CutCellDG.cell_sign(mesh, cellid)
        @assert s == -1 || s == 0 || s == 1
        if s == 1 || s == 0
            cellmap = CutCellDG.cell_map(mesh, +1, cellid)
            pquad = cellquads[+1, cellid]
            add_cell_norm_squared!(vals, func, cellmap, pquad)
        end
        if s == -1 || s == 0
            cellmap = CutCellDG.cell_map(mesh, -1, cellid)
            nquad = cellquads[-1, cellid]
            add_cell_norm_squared!(vals, func, cellmap, nquad)
        end
    end
    return sqrt.(vals)
end

function uniform_mesh_L2_error(nodalsolutions, exactsolution, basis, quad, mesh)
    ndofs, nnodes = size(nodalsolutions)
    err = zeros(ndofs)
    interpolater = InterpolatingPolynomial(ndofs, basis)
    ncells = CutCellDG.number_of_cells(mesh)

    for cellid = 1:ncells
        cellmap = CutCellDG.cell_map(mesh, cellid)
        nodeids = CutCellDG.nodal_connectivity(mesh, cellid)
        elementsolution = nodalsolutions[:, nodeids]
        update!(interpolater, elementsolution)
        add_cell_error_squared!(err, interpolater, exactsolution, cellmap, quad)
    end
    return sqrt.(err)
end

function integral_norm_on_uniform_mesh(func, quad, mesh, ndofs)
    numcells = CutCellDG.number_of_cells(mesh)
    vals = zeros(ndofs)
    for cellid = 1:numcells
        cellmap = CutCellDG.cell_map(mesh, cellid)
        add_cell_norm_squared!(vals, func, cellmap, quad)
    end
    return sqrt.(vals)
end
