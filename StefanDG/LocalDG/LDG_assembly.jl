function assemble_LDG_linear_system!(
    sysmatrix,
    solverbasis,
    cellquads,
    facequads,
    interfacequads,
    k1,
    k2,
    penaltyfactor,
    beta,
    mesh,
)
    ############################################################################
    # LINEAR SYSTEM ASSEMBLY
    LocalDG.assemble_divergence_operator!(
        sysmatrix,
        solverbasis,
        cellquads,
        mesh,
    )
    LocalDG.assemble_mass_operator!(sysmatrix, solverbasis, cellquads, mesh)
    LocalDG.assemble_gradient_operator!(
        sysmatrix,
        solverbasis,
        cellquads,
        k1,
        k2,
        mesh,
    )
    ############################################################################

    ############################################################################
    # INTERELEMENT CONDITION
    LocalDG.assemble_interelement_scalar_flux_operator!(
        sysmatrix,
        solverbasis,
        facequads,
        beta,
        mesh,
    )
    LocalDG.assemble_interelement_vector_flux_operator!(
        sysmatrix,
        solverbasis,
        facequads,
        k1,
        k2,
        penaltyfactor,
        beta,
        mesh,
    )
    ############################################################################

    ############################################################################
    # INTERFACE CONDITION
    LocalDG.assemble_interface_scalar_flux_operator!(
        sysmatrix,
        solverbasis,
        interfacequads,
        beta,
        mesh,
    )
    LocalDG.assemble_interface_vector_flux_operator!(
        sysmatrix,
        solverbasis,
        interfacequads,
        k1,
        k2,
        penaltyfactor,
        beta,
        mesh,
    )
    ############################################################################

    ############################################################################
    # BOUNDARY CONDITIONS
    LocalDG.assemble_boundary_vector_flux_operator!(
        sysmatrix,
        solverbasis,
        facequads,
        k1,
        k2,
        penaltyfactor,
        mesh,
    )
    ############################################################################
end

function assemble_LDG_rhs!(
    sysrhs,
    sourceterm,
    boundaryfunc,
    solverbasis,
    cellquads,
    facequads,
    penaltyfactor,
    mesh,
)

    ################################################################################
    # BOUNDARY CONDITION
    LocalDG.assemble_boundary_scalar_flux_source!(
        sysrhs,
        boundaryfunc,
        solverbasis,
        facequads,
        mesh,
    )
    LocalDG.assemble_boundary_vector_flux_source!(
        sysrhs,
        boundaryfunc,
        solverbasis,
        facequads,
        penaltyfactor,
        mesh,
    )
    ################################################################################

    ################################################################################
    # SOURCE TERM
    LocalDG.assemble_source!(sysrhs,sourceterm,solverbasis,cellquads,mesh)
    ################################################################################
end
