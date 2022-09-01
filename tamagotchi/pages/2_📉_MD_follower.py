############################################################################################
# # # # # # # # # # # # # # # # # #     MD FOLLOWER      # # # # # # # # # # # # # # # # # #
############################################################################################

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

ss = st.session_state

options = [out for out in ss.OUTs] + [xyz for xyz in ss.XYZs]
options.sort(key=lambda x: x.name)
selections = st.sidebar.multiselect(
    "Select output files", options, format_func=lambda x: x.name
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


tab1, tab2, tab3 = st.tabs(
    [
        "Energy",
        "STD",
        "Volume/Pressure",
    ]
)

with tab1:

    # local_options = [item for item in Path(".").glob("**/*md.out")]

    for file in selections:

        if file.name[-4:] == ".out":
            df = read_md_out(file)
        elif file.name[-4:] == ".xyz":
            df = read_xyz_traj(file)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        title = file.name

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

        # fig.add_trace(
        #    go.Scatter(
        #        x=df.index*ss.timestep,
        #        y=df["Total MD Energy"].expanding().mean()*23.06,
        #        name="Expanding Average",
        #        line={
        #            "width": 3,
        #            "color": "turquoise",
        #            },
        #    ),
        # )
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
            key=f"{file.name}_en",
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

        # fig.add_trace(
        #    go.Scatter(
        #        x=df.index[int(start/ss.timestep/ss.mdrestartfreq):int(stop/ss.timestep/ss.mdrestartfreq)]*ss.timestep,
        #        y=df["Total MD Energy"].iloc()[int(start/ss.timestep/ss.mdrestartfreq):int(stop/ss.timestep/ss.mdrestartfreq):].expanding().mean()*23.06,
        #        name=f"Expanding Average between {start} and {stop} ps",
        #        line={
        #            "width": 3,
        #            "color": "purple",
        #            },
        #    ),
        # )
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


with tab2:

    fig2_bins = st.number_input(
        label="Number of bins:",
        value=100,
    )

    for file in selections:

        if file.name[-4:] == ".out":
            df = read_md_out(file)
        elif file.name[-4:] == ".xyz":
            df = read_xyz_traj(file)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        title = file.name

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
            key=f"{file.name}_dist",
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


with tab3:

    for file in selections:

        if file.name[-4:] == ".out":
            df = read_md_out(file)
        elif file.name[-4:] == ".xyz":
            continue

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        title = file.name

        fig.update_layout(title=title)

        fig.add_trace(
            go.Scatter(
                x=df.index * ss.timestep,
                y=df["Volume"],
                name="Volume",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index * ss.timestep,
                y=df["Pressure"] / 101325,
                name="Pressure",
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Volume (A^3)", row=2, secondary_y=False)
        fig.update_yaxes(title_text="Pressure (atm)", row=2, secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
