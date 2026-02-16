import numpy as np


def aspect_parameter(cpnt_names):
    Compositional_heating_values = []

    Thermal_diffusivities = []
    Heat_capacities = []
    Densities = []
    Thermal_expansivities = []

    Prefactors_dfc = []
    Stress_exponents_dfc = []
    Grain_size_exponents_dfc = []
    Activation_energies_dfc = []
    Activation_volumes_dfc = []

    Prefactors_dsc = []
    Stress_exponents_dsc = []
    Activation_energies_dsc = []
    Activation_volumes_dsc = []

    Angles_internal_friction = []
    Cohesions = []

    for cpnt_name in cpnt_names:
        if cpnt_name in Compositional_heating_values_dict:
            Compositional_heating_values.append(Compositional_heating_values_dict[cpnt_name])

        Thermal_diffusivities.append(Thermal_diffusivities_dict[cpnt_name])
        Heat_capacities.append(Heat_capacities_dict[cpnt_name])
        Densities.append(Densities_dict[cpnt_name])
        Thermal_expansivities.append(Thermal_expansivities_dict[cpnt_name])

        Prefactors_dfc.append(Prefactors_dfc_dict[cpnt_name])
        Stress_exponents_dfc.append(Stress_exponents_dfc_dict[cpnt_name])
        Grain_size_exponents_dfc.append(Grain_size_exponents_dfc_dict[cpnt_name])
        Activation_energies_dfc.append(Activation_energies_dfc_dict[cpnt_name])
        Activation_volumes_dfc.append(Activation_volumes_dfc_dict[cpnt_name])

        Prefactors_dsc.append(Prefactors_dsc_dict[cpnt_name])
        Stress_exponents_dsc.append(Stress_exponents_dsc_dict[cpnt_name])
        Activation_energies_dsc.append(Activation_energies_dsc_dict[cpnt_name])
        Activation_volumes_dsc.append(Activation_volumes_dsc_dict[cpnt_name])

        Angles_internal_friction.append(Angles_internal_friction_dict[cpnt_name])
        Cohesions.append(Cohesions_dict[cpnt_name])

    def format_list(lst, width=16):
        # return ", ".join(f"{x:.3e}".rjust(width) for x in lst)
        return ", ".join(f"{x}".rjust(width) for x in lst)


    output = "# Dry olivine flow law for lithospheric mantle and sub-lithospheric mantle(CP & OP)\n"

    output += f"set Compositional heating values = {format_list(Compositional_heating_values)}\n"

    output += f"set Thermal diffusivities = {format_list(Thermal_diffusivities)}\n"
    output += f"set Heat capacities       = {format_list(Heat_capacities, width=12)}\n"
    output += f"set Densities             = {format_list(Densities, width=12)}\n"
    output += f"set Thermal expansivities = {format_list(Thermal_expansivities)}\n"

    output += f"set Prefactors for diffusion creep           = {format_list(Prefactors_dfc)}\n"
    output += f"set Stress exponents for diffusion creep     = {format_list(Stress_exponents_dfc, width=12)}\n"
    output += f"set Grain size exponents for diffusion creep = {format_list(Grain_size_exponents_dfc, width=12)}\n"
    output += f"set Activation energies for diffusion creep  = {format_list(Activation_energies_dfc)}\n"
    output += f"set Activation volumes for diffusion creep   = {format_list(Activation_volumes_dfc)}\n"

    output += f"set Prefactors for dislocation creep          = {format_list(Prefactors_dsc)}\n"
    output += f"set Stress exponents for dislocation creep    = {format_list(Stress_exponents_dsc, width=12)}\n"
    output += f"set Activation energies for dislocation creep = {format_list(Activation_energies_dsc)}\n"
    output += f"set Activation volumes for dislocation creep  = {format_list(Activation_volumes_dsc)}\n"

    output += f"set Angles of internal friction               = {format_list(Angles_internal_friction, width=12)}\n"
    output += f"set Cohesions                                 = {format_list(Cohesions)}\n"

    with open('parameters.txt', 'w') as file:
        file.write(output)

Compositional_heating_values_dict = {'upper mantle': 0., 'plasticstrain': 0., 'cuc': 1.0e-6,
                              'clc': 4.e-7, 'cm': 0., 'sediment': 0.,
                              'oc': 0., 'om': 0., 'wz': 0., 'lowermantle':0.}

Thermal_diffusivities_dict = {'upper mantle': 9.891197e-7, 'plasticstrain': 9.891197e-7, 'cuc': 1.207297e-6,
                              'clc': 1.149425e-6, 'cm': 9.891197e-7, 'sediment': 1.207297e-6,
                              'oc': 1.149425e-6, 'om': 9.891197e-7, 'wz': 1.207297e-6, 'lowermantle': 9.891197e-7}

Heat_capacities_dict = {'upper mantle': 1250, 'plasticstrain': 1250, 'cuc': 750, 'clc': 750, 'cm': 1250,
                        'sediment': 750, 'oc': 750, 'om': 1250, 'wz': 750, 'lowermantle': 1250}

# Densities_dict = {'upper mantle': 3370, 'plasticstrain': 3370, 'cuc': 2800, 'clc': 2900, 'cm': 3370,
#                   'sediment': 3000, 'oc': 3000, 'om': 3370, 'wz': 3300, 'lowermantle': 3370}
Densities_dict = {'upper mantle': 3370, 'plasticstrain': 3370, 'cuc': 2800, 'clc': 2900, 'cm': 3370,
                  'sediment': 3300, 'oc': 3300, 'om': 3370, 'wz': 3300, 'lowermantle': 3370}
# 参考prm中sediment为3300，oc为3300.
Thermal_expansivities_dict = {'upper mantle': 3e-5, 'plasticstrain': 3e-5, 'cuc': 2.7e-5, 'clc': 2.7e-5,
                              'cm': 3e-5, 'sediment': 2.7e-5, 'oc': 2.7e-5, 'om': 3e-5, 'wz': 2.7e-5,
                              'lowermantle': 3e-5}

Prefactors_dfc_dict = {'upper mantle': 2.37e-15, 'plasticstrain': 1.e-50, 'cuc': 1.e-50, 'clc': 1.e-50, 'cm': 2.37e-15,
                       'sediment': 1.e-50, 'oc': 1.e-50, 'om': 2.37e-15, 'wz': 1.e-50, 'lowermantle': 1.67e-13}

Stress_exponents_dfc_dict = {'upper mantle': 1.0, 'plasticstrain': 1.0, 'cuc': 1.0, 'clc': 1.0, 'cm': 1.0,
                             'sediment': 1.0, 'oc': 1.0, 'om': 1.0, 'wz': 1.0, 'lowermantle': 1.0}

Grain_size_exponents_dfc_dict = {'upper mantle': 3., 'plasticstrain': 0, 'cuc': 1., 'clc': 1., 'cm': 3.,
                                 'sediment': 1., 'oc': 1., 'om': 3.0, 'wz': 1.0, 'lowermantle': 1.}

Activation_energies_dfc_dict = {'upper mantle': 375.e3, 'plasticstrain': 0, 'cuc': 0, 'clc': 0, 'cm': 375.e3,
                                'sediment': 0, 'oc': 0, 'om': 375.e3, 'wz': 0, 'lowermantle': 150.e3}

Activation_volumes_dfc_dict = {'upper mantle': 4.e-6, 'plasticstrain': 0., 'cuc': 0., 'clc': 0., 'cm': 4.e-6,
                               'sediment': 0., 'oc': 0., 'om': 4.e-6, 'wz': 0., 'lowermantle': 8.34e-7}

Prefactors_dsc_dict = {'upper mantle': 6.52e-16, 'plasticstrain': 6.52e-16, 'cuc': 8.57e-28, 'clc': 7.13e-18, 'cm': 6.52e-16,
                       'sediment': 1.12e-10, 'oc': 1.12e-10, 'om': 6.52e-16, 'wz': 1.12e-10, 'lowermantle': 6.51e-16}

Stress_exponents_dsc_dict = {'upper mantle': 3.5, 'plasticstrain': 3.5, 'cuc': 4.0, 'clc': 3.0, 'cm': 3.5,
                             'sediment': 3.4, 'oc': 3.4, 'om': 3.5, 'wz': 3.4, 'lowermantle': 3.5}

Activation_energies_dsc_dict = {'upper mantle': 530.e3, 'plasticstrain': 530.e3, 'cuc': 223.e3, 'clc': 345.e3, 'cm': 530.e3,
                                'sediment': 497.e3, 'oc': 497.e3, 'om': 530.e3, 'wz': 497.e3, 'lowermantle': 530.e3}

Activation_volumes_dsc_dict = {'upper mantle': 18.e-6, 'plasticstrain': 18.e-6, 'cuc': 0., 'clc': 0., 'cm': 18.e-6,
                               'sediment': 0., 'oc': 0., 'om': 18.e-6, 'wz': 0., 'lowermantle': 1.3e-5}

Angles_internal_friction_dict = {'upper mantle': 30.00, 'plasticstrain': 30.00, 'cuc': 30.00, 'clc': 30.00, 'cm': 30.00,
                                 'sediment': 2.00, 'oc': 2.00, 'om': 30.00, 'wz': 2.00, 'lowermantle': 1.434}

Cohesions_dict = {'upper mantle': 20.e6, 'plasticstrain': 20.e6, 'cuc': 20.e6, 'clc': 20.e6, 'cm': 20.e6,
                  'sediment': 10.e6, 'oc': 10.e6, 'om': 20.e6, 'wz': 10.e6, 'lowermantle': 4.e6}

# component_names = ['upper mantle', 'cuc', 'clc', 'cm', 'sediment', 'oc', 'om', 'wz' ,'cuc', 'clc', 'cm', 'cuc', 'clc', 'cm', 'wz', 'sediment', 'oc', 'om', 'wz', 'sediment', 'oc', 'om', 'plasticstrain']
# component_names = ['upper mantle', 'plasticstrain', 'cuc', 'clc', 'cm', 'sediment', 'oc', 'om', 'wz' ,'cuc', 'clc', 'cm', 'sediment', 'oc', 'om', 'wz', 'sediment', 'oc', 'om', 'wz', 'sediment', 'oc', 'om']
# t13
component_names = ['upper mantle', 'cuc', 'clc', 'cm', 'sediment', 'oc', 'om', 'wz' ,'cuc', 'clc', 'cm', 'cuc', 'clc', 'cm', 'wz', 'sediment', 'oc', 'om', 'wz', 'sediment', 'oc', 'om', 'plasticstrain']
#　        # mantle, luc, llc, lmc, lsed, loc, lom, lwk, muc, mlc, mmc, scuc, sclc, scmc, rwk, rsed, roc, rom, rrwk, rrsed, rroc, rrmc, plastic_strain
aspect_parameter(component_names)
print('component_length:', len(component_names))