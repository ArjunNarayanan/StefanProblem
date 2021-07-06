import Base.==, Base.≈

function allapprox(v1, v2)
    return all(v1 .≈ v2)
end

function allapprox(v1, v2, atol)
    err = maximum(abs.(v1 - v2))
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

function uniform_mesh_L2_error(nodalsolutions, exactsolution, basis, quad, mesh)
    ndofs, nnodes = size(nodalsolutions)
    err = zeros(ndofs)
    interpolater = InterpolatingPolynomial(ndofs, basis)
    ncells = DG1D.number_of_cells(mesh)

    for cellid = 1:ncells
        cellmap = DG1D.cell_map(mesh, cellid)
        nodeids = DG1D.nodal_connectivity(mesh, cellid)
        elementsolution = nodalsolutions[:, nodeids]
        update!(interpolater, elementsolution)
        add_cell_error_squared!(err, interpolater, exactsolution, cellmap, quad)
    end
    return sqrt.(err)
end

function integral_norm_on_uniform_mesh(func, quad, mesh, ndofs)
    numcells = DG1D.number_of_cells(mesh)
    vals = zeros(ndofs)
    for cellid = 1:numcells
        cellmap = DG1D.cell_map(mesh, cellid)
        add_cell_norm_squared!(vals, func, cellmap, quad)
    end
    return sqrt.(vals)
end
