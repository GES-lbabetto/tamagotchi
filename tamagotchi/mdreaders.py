import pandas as pd
import streamlit as st

ss = st.session_state

"""
### NOTE ###
Units are as follows:
  - length: Å
  - temperature: K
  - pressure: Pa
  - energy: eV
"""


def read_dftb_out(dftb_out):

    step = 0

    steps = []
    md_steps = []
    volume = []
    pressures = []
    gibbs_free_energies = []
    gibbs_free_energies_including_ke = []

    potential_energies = []
    kinetic_energies = []
    total_md_energies = []
    temperatures = []

    md_step_point = None
    volume_point = None
    pressures_point = None
    gibbs_free_energy_point = None
    gibbs_free_energy_including_ke_point = None
    potential_energies_point = None
    kinetic_energies_point = None
    total_md_energy_point = None
    temperature_point = None

    for line in dftb_out:

        if "MD step" in line:
            md_step_point = int(line.split()[2])
        if "Volume:" in line:
            volume_point = float(line.split()[3])  # A^3
        if "Pressure:" in line:
            pressures_point = float(line.split()[3])  # Pa
        if "Gibbs free energy:" in line:
            gibbs_free_energy_point = float(line.split()[5])  # eV
        if "Gibbs free energy including KE:" in line:
            gibbs_free_energy_including_ke_point = float(line.split()[7])  # eV
        if "Potential Energy:" in line:
            potential_energies_point = float(line.split()[4])  # eV
        if "MD Kinetic Energy:" in line:
            kinetic_energies_point = float(line.split()[5])  # eV
        if "Total MD Energy:" in line:
            total_md_energy_point = float(line.split()[5])  # eV
        if "MD Temperature:" in line:
            temperature_point = float(line.split()[4])  # K

            steps.append(step)
            step += ss.mdrestartfreq

            md_steps.append(md_step_point)
            volume.append(volume_point)
            pressures.append(pressures_point)
            gibbs_free_energies.append(gibbs_free_energy_point)
            gibbs_free_energies_including_ke.append(gibbs_free_energy_including_ke_point)
            potential_energies.append(potential_energies_point)
            kinetic_energies.append(kinetic_energies_point)
            total_md_energies.append(total_md_energy_point)
            temperatures.append(temperature_point)

    df = pd.DataFrame(
        {
            "Volume": volume,
            "Pressure": pressures,
            "Gibbs Free Energy": gibbs_free_energies,
            "Gibbs Free Energies including KE": gibbs_free_energies,
            "Potential Energy": potential_energies,
            "MD Kinetic Energy": kinetic_energies,
            "Total MD Energy": total_md_energies,
            "MD Temperature": temperatures,
        },
        # index=md_steps,
        index=steps,
    )

    return df


def read_xyz_traj(xyz_traj):

    step = 0

    steps = []
    md_steps = []
    volume = []
    pressures = []
    gibbs_free_energies = []
    gibbs_free_energies_including_ke = []

    potential_energies = []
    kinetic_energies = []
    total_md_energies = []
    temperatures = []

    md_step_point = None
    volume_point = None
    pressures_point = None
    gibbs_free_energy_point = None
    gibbs_free_energy_including_ke_point = None
    potential_energies_point = None
    kinetic_energies_point = None
    total_md_energy_point = None
    temperature_point = None

    for line in xyz_traj:

        if "Step" in line:
            md_step_point = int(line.split()[1])
            total_md_energy_point = float(line.split()[3]) * 27.2114  # eV

            steps.append(step)
            step += ss.mdrestartfreq

            md_steps.append(md_step_point)
            total_md_energies.append(total_md_energy_point)

    df = pd.DataFrame(
        {
            # "Volume": volume,
            # "Pressure": pressures,
            # "Gibbs Free Energy": gibbs_free_energies,
            # "Gibbs Free Energies including KE": gibbs_free_energies,
            # "Potential Energy": potential_energies,
            # "MD Kinetic Energy": kinetic_energies,
            "Total MD Energy": total_md_energies,
            # "MD Temperature": temperatures,
        },
        # index=md_steps,
        index=steps,
    )

    return df


def read_namd_out(namd_out):

    step = 0

    steps = []
    md_steps = []
    volume = []
    pressures = []
    gibbs_free_energies = []
    gibbs_free_energies_including_ke = []

    potential_energies = []
    kinetic_energies = []
    total_md_energies = []
    temperatures = []

    md_step_point = None
    volume_point = None
    pressures_point = None
    gibbs_free_energy_point = None
    gibbs_free_energy_including_ke_point = None
    potential_energies_point = None
    kinetic_energies_point = None
    total_md_energy_point = None
    temperature_point = None

    for line in namd_out:

        if "ENERGY:  " in line:
            md_step_point = int(line.split()[1])
            volume_point = float(line.split()[18])  # A^3
            pressures_point = float(line.split()[16]) * 1e5  # Pa
            # gibbs_free_energy_point = float(line.split()[5])  # eV
            # gibbs_free_energy_including_ke_point = float(line.split()[7])  # eV
            potential_energies_point = float(line.split()[13]) / 23.06  # eV
            kinetic_energies_point = float(line.split()[10]) / 23.06  # eV
            total_md_energy_point = float(line.split()[11]) / 23.06  # eV
            temperature_point = float(line.split()[12]) / 23.06  # K

            steps.append(step)
            step += ss.mdrestartfreq

            md_steps.append(md_step_point)
            volume.append(volume_point)
            pressures.append(pressures_point)
            # gibbs_free_energies.append(gibbs_free_energy_point)
            # gibbs_free_energies_including_ke.append(gibbs_free_energy_including_ke_point)
            potential_energies.append(potential_energies_point)
            kinetic_energies.append(kinetic_energies_point)
            total_md_energies.append(total_md_energy_point)
            temperatures.append(temperature_point)

    df = pd.DataFrame(
        {
            "Volume": volume,
            "Pressure": pressures,
            # "Gibbs Free Energy": gibbs_free_energies,
            # "Gibbs Free Energies including KE": gibbs_free_energies,
            "Potential Energy": potential_energies,
            "MD Kinetic Energy": kinetic_energies,
            "Total MD Energy": total_md_energies,
            "MD Temperature": temperatures,
        },
        index=md_steps,
        # index=steps,
    )

    return df
