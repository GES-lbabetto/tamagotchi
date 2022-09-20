############################################################################################
# # # # # # # # # # # # # # # # # #     MD FOLLOWER      # # # # # # # # # # # # # # # # # #
############################################################################################

from functools import total_ordering
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

ss = st.session_state


options = [ss.MDs[md] for md in ss.MDs]
options.sort(key=lambda x: x.name)
selections = st.sidebar.multiselect(
    "Select MD experiments", options, format_func=lambda x: x.name
)


@st.cache
def read_md_out(md_out):

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

    for line in md_out:

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


@st.cache
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


energy_tab, std_tab, density_tab = st.tabs(
    [
        "📉 Energy",
        "📊 STD",
        "📈 Density",
    ]
)

with energy_tab:

    for md in selections:

        if md.out:
            df = read_md_out(md.out)
        elif md.xyz:
            df = read_xyz_traj(md.xyz)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        title = md.name

        fig.update_layout(title=title)

        fig.add_trace(
            go.Scatter(
                x=df.index * ss.timestep,
                y=df["Total MD Energy"] * 23.06,
                name="Total MD Energy",
                line={
                    "width": 0.1,
                    "color": "blue",
                },
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=df.index * ss.timestep,
                y=df["Total MD Energy"]
                .rolling(window=int(10 / ss.timestep / ss.mdrestartfreq))
                .mean()
                * 23.06,
                name="Rolling Average on 10 ps",
                line={
                    "width": 1,
                    "color": "blue",
                },
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=df.index[::-1] * ss.timestep,
                y=df["Total MD Energy"].iloc()[::-1].expanding().mean() * 23.06,
                name="Reverse Expanding Average",
                line={
                    "width": 3,
                    "color": "orange",
                },
            ),
        )
        fig.update_xaxes(title_text="time (ps)")
        fig.update_yaxes(title_text="Energy (kcal/mol)")

        start, stop = st.slider(
            label="Calculate average energy from/to (ps):",
            min_value=int(df.index[0] * ss.timestep),
            max_value=int(df.index[-1] * ss.timestep),
            value=(int(df.index[0] * ss.timestep), int(df.index[-1] * ss.timestep)),
            key=f"{md.name}_en",
        )

        average = (
            df["Total MD Energy"]
            .iloc()[
                int(start / ss.timestep / ss.mdrestartfreq) : int(
                    stop / ss.timestep / ss.mdrestartfreq
                ) :
            ]
            .mean()
        )
        dataset_std = (
            df["Total MD Energy"]
            .iloc()[
                int(start / ss.timestep / ss.mdrestartfreq) : int(
                    stop / ss.timestep / ss.mdrestartfreq
                ) :
            ]
            .std()
        )
        npoints = (
            stop / ss.timestep / ss.mdrestartfreq - start / ss.timestep / ss.mdrestartfreq
        )
        average_error = dataset_std / np.sqrt((npoints))
        st.write(
            f"**Average Energy between {start} and {stop} ps:**\n"
            f"* {average:.{len(f'{average_error:.2}')-2}f} ± {average_error:.2} eV\n"
            f"* {average*23.06:.{len(f'{average_error*23.06:.2}')-2}f} ± {average_error*23.06:.2} kcal/mol"
        )

        fig.add_trace(
            go.Scatter(
                x=df.index[
                    int(start / ss.timestep / ss.mdrestartfreq) : int(
                        stop / ss.timestep / ss.mdrestartfreq
                    )
                ]
                * ss.timestep,
                y=[average * 23.06] * len(df.index),
                name=f"Average energy between {start} and {stop} ps",
                line={
                    "width": 3,
                    "color": "black",
                    "dash": "dash",
                },
            )
        )
        fig.update_xaxes(range=[start, stop])
        fig.update_yaxes(
            range=[
                df["Total MD Energy"]
                .rolling(window=int(10 / ss.timestep / ss.mdrestartfreq))
                .mean()
                .iloc()[
                    int(start / ss.timestep / ss.mdrestartfreq) : int(
                        stop / ss.timestep / ss.mdrestartfreq
                    ) :
                ]
                .min()
                * 23.06
                * (1 + 1e-5),
                df["Total MD Energy"]
                .rolling(window=int(10 / ss.timestep / ss.mdrestartfreq))
                .mean()
                .iloc()[
                    int(start / ss.timestep / ss.mdrestartfreq) : int(
                        stop / ss.timestep / ss.mdrestartfreq
                    ) :
                ]
                .max()
                * 23.06
                * (1 - 1e-5),
            ]
        )
        st.plotly_chart(fig, use_container_width=True)


with std_tab:

    fig2_bins = st.number_input(
        label="Number of bins:",
        value=100,
    )

    for md in selections:

        if md.out:
            df = read_md_out(md.out)
        elif md.xyz:
            df = read_xyz_traj(md.xyz)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        title = md.name

        fig.update_layout(title=title)

        fig.add_trace(
            go.Scatter(
                x=df.index * ss.timestep,
                y=df["Total MD Energy"].expanding().mean() * 23.06,
                name="Total MD Energy Average",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index * ss.timestep,
                y=df["Total MD Energy"]
                .expanding()
                .mean()
                .rolling(window=int(10 / ss.timestep / ss.mdrestartfreq))
                .std()
                * 23.06,
                name="Standard Deviation on 10 ps window",
            ),
            secondary_y=True,
        )
        fig.update_xaxes(title_text="time (ps)")
        fig.update_yaxes(
            title_text="Energy (kcal/mol)",
            range=[
                df["Total MD Energy"]
                .expanding()
                .mean()
                .iloc()[int(df.index[-1] / 5 / ss.mdrestartfreq) : :]
                .min()
                * 23.06
                * (1 + 1e-5),
                df["Total MD Energy"]
                .expanding()
                .mean()
                .iloc()[int(df.index[-1] / 5 / ss.mdrestartfreq) : :]
                .max()
                * 23.06
                * (1 - 1e-5),
            ],
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Standard deviation (kcal/mol)",
            range=[0, 1],
            secondary_y=True,
        )

        fig2_start, fig2_stop = st.slider(
            label="Calculate energy distribution from/to (ps):",
            min_value=int(df.index[0] * ss.timestep),
            max_value=int(df.index[-1] * ss.timestep),
            value=(int(df.index[0] * ss.timestep), int(df.index[-1] * ss.timestep)),
            key=f"{md.name}_dist",
        )
        fig2 = go.Figure(
            px.histogram(
                x=df["Total MD Energy"].iloc()[
                    int(fig2_start / ss.timestep / ss.mdrestartfreq) : int(
                        fig2_stop / ss.timestep / ss.mdrestartfreq
                    ) :
                ],
                nbins=int(fig2_bins),
            )
        )
        fig2.update_layout(title=f"{title} - Distribution of energies")
        fig2.update_xaxes(title_text="Energy (eV)")
        fig2.update_yaxes(title_text="Counts")

        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)


with density_tab:

    amu = 1.66054e-24
    atom_weights = {
        "H": 1.008,
        "Li": 6.94,
        "Be": 9.012,
        "B": 10.81,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "F": 18.998,
        "Na": 22.989,
        "Mg": 24.305,
        "Al": 26.981,
        "Si": 28.085,
        "P": 30.973,
        "S": 32.06,
        "Cl": 35.45,
        "K": 39.0983,
        "Ca": 40.078,
        "Sc": 44.955,
        "Ti": 47.867,
        "V": 50.9415,
        "Cr": 51.9961,
        "Mn": 54.938,
        "Fe": 55.845,
        "Co": 58.933,
        "Ni": 58.6934,
        "Cu": 63.546,
        "Zn": 65.38,
        "Ga": 69.723,
        "Ge": 72.630,
        "As": 74.921,
        "Se": 78.971,
        "Br": 79.904,
        "Pd": 106.42,
        "Ag": 107.8682,
        "Cd": 112.414,
        "Sn": 118.710,
        "I": 126.904,
        "Pt": 195.084,
        "Au": 196.966,
        "Hg": 200.592,
    }

    for md in selections:

        if md.out:
            df = read_md_out(md.out)
        else:
            st.warning(f"Sorry, no trajectory file available for {md.name}!")
            continue

        total_weight = 0.0
        if md.mol2:
            for line in md.mol2:
                if len(line.split()) == 9:
                    total_weight += float(atom_weights[line.split()[1]])
            total_weight *= amu  # g
        elif md.pbd:
            for line in md.pbd:
                if len(line.split()) == 11:
                    total_weight += float(atom_weights[line.split()[2]])
            total_weight *= amu  # g
        else:
            st.warning(f"Sorry, no topology file available for {md.name}!")
            continue

        dens_ps = st.slider(
            label="Calculate density at (ps):",
            min_value=int(df.index[0] * ss.timestep),
            max_value=int(df.index[-1] * ss.timestep),
            value=int(int(df.index[-1] * ss.timestep)),
            key=f"{md.name}_dens",
        )

        sel_dens = (
            total_weight
            / df["Volume"].iloc()[int(dens_ps / ss.timestep / ss.mdrestartfreq)]
            / 1e-27
        )
        st.write(f"Density after {dens_ps:.0f} ps: **{sel_dens:.2f}** g/L\n")

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        title = md.name

        fig.update_layout(title=title)

        fig.add_trace(
            go.Scatter(
                x=df.index * ss.timestep,
                y=df["Pressure"] / 101325,
                name="Pressure",
                line={
                    "width": 0.5,
                },
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index * ss.timestep,
                y=total_weight / (df["Volume"] * 1e-27),
                name="Density",
                line={
                    "width": 3,
                },
            ),
            secondary_y=True,
        )

        fig.update_xaxes(title_text="time (ps)")
        fig.update_yaxes(title_text="Pressure (atm)", secondary_y=False)
        fig.update_yaxes(title_text="Density (g/L)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
