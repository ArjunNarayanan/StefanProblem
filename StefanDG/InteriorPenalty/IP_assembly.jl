function assemble_interior_penalty_linear_system!(
    sysmatrix,
    solverbasis,
    cellquads,
    facequads,
    interfacequads,
    k1,
    k2,
    penalty,
    mesh,
)

    ################################################################################
    # LINEAR SYSTEM
    InteriorPenalty.assemble_gradient_operator!(
        sysmatrix,
        solverbasis,
        cellquads,
        k1,
        k2,
        mesh,
    )
    ################################################################################

    ################################################################################
    # INTERELEMENT CONDITION
    InteriorPenalty.assemble_interelement_flux_operator!(
        sysmatrix,
        solverbasis,
        facequads,
        k1,
        k2,
        mesh,
    )
    InteriorPenalty.assemble_interelement_penalty_operator!(
        sysmatrix,
        solverbasis,
        facequads,
        penalty,
        mesh,
    )
    ################################################################################

    ################################################################################
    # INTERFACE CONDITION
    InteriorPenalty.assemble_interface_flux_operator!(
        sysmatrix,
        solverbasis,
        interfacequads,
        k1,
        k2,
        mesh,
    )
    InteriorPenalty.assemble_interface_penalty_operator!(
        sysmatrix,
        solverbasis,
        interfacequads,
        penalty,
        mesh,
    )
    ################################################################################

    ################################################################################
    # BOUNDARY CONDITIONS
    InteriorPenalty.assemble_boundary_flux_operator!(
        sysmatrix,
        solverbasis,
        facequads,
        k1,
        k2,
        mesh,
    )
    InteriorPenalty.assemble_boundary_penalty_operator!(
        sysmatrix,
        solverbasis,
        facequads,
        penalty,
        mesh,
    )
end

function assemble_interior_penalty_rhs!(
    sysrhs,
    sourceterm,
    boundaryfunc,
    solverbasis,
    cellquads,
    facequads,
    penalty,
    mesh,
)
    InteriorPenalty.assemble_boundary_source!(
        sysrhs,
        boundaryfunc,
        solverbasis,
        facequads,
        penalty,
        mesh,
    )
    ################################################################################
    # SOURCE TERM
    InteriorPenalty.assemble_source!(
        sysrhs,
        sourceterm,
        solverbasis,
        cellquads,
        mesh,
    )
    ################################################################################

end
