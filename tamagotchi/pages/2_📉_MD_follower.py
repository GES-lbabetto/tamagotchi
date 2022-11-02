############################################################################################
# # # # # # # # # # # # # # # # # #     MD FOLLOWER      # # # # # # # # # # # # # # # # # #
############################################################################################

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import mdreaders

ss = st.session_state


options = [ss.MDs[md] for md in ss.MDs]
options.sort(key=lambda x: x.name)
selections = st.sidebar.multiselect(
    "Select MD experiments", options, format_func=lambda x: x.name
)

# Importing file readers


@st.cache
def read_dftb_out(dftb_out):
    df = mdreaders.read_dftb_out(dftb_out)
    return df


@st.cache
def read_xyz_traj(xyz_traj):
    df = mdreaders.read_xyz_traj(xyz_traj)
    return df


@st.cache
def read_namd_out(namd_out):
    df = mdreaders.read_namd_out(namd_out)
    return df


def load_output(md):
    try:
        df = read_dftb_out(md)
        st.success(f"DFTB output found for {md.name}!", icon="✔")
    except AttributeError:
        try:
            df = read_namd_out(md)
            st.success(f"NAMD output found for {md.name}!", icon="✔")
        except AttributeError:
            try:
                df = read_xyz_traj(md)
                st.success(f"XYZ trajectory found for {md.name}!", icon="✔")
            except AttributeError:
                st.error(f"No trajectory file found for {md.name}!", icon="❌")
                st.stop()
    return df


energy_tab, convergence_tab, density_tab = st.tabs(
    [
        "📉 Energy",
        "⌚ Convergence",
        "📈 Density",
    ]
)

with energy_tab:

    for md in selections:

        df = load_output(md)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        title = md.name

        fig.update_layout(title=title)

        fig.add_trace(
            go.Scatter(
                x=df.index * (md.timestep / 1000),
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
                x=df.index * (md.timestep / 1000),  # ps
                y=df["Total MD Energy"]
                .rolling(window=int(10 / (md.timestep / 1000) / md.stride))
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
                x=df.index[::-1] * md.timestep / 1000,  # ps
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
            min_value=int(df.index[0] * md.timestep / 1000),
            max_value=int(df.index[-1] * md.timestep / 1000),
            value=(
                int(df.index[0] * md.timestep / 1000),
                int(df.index[-1] * md.timestep / 1000),
            ),
            key=f"{md.name}_en",
        )

        fig.add_trace(
            go.Scatter(
                x=df.index[
                    int(start / (md.timestep / 1000) / md.stride) : int(
                        stop / (md.timestep / 1000) / md.stride
                    ) :
                ]
                * md.timestep
                / 1000,
                y=df["Total MD Energy"]
                .iloc()[
                    int(start / (md.timestep / 1000) / md.stride) : int(
                        stop / (md.timestep / 1000) / md.stride
                    ) :
                ]
                .expanding()
                .mean()
                * 23.06,
                name=f"Expanding Average between {start} and {stop} ps",
                line={
                    "width": 3,
                    "color": "red",
                },
            ),
        )

        average = (
            df["Total MD Energy"]
            .iloc()[
                int(start / (md.timestep / 1000) / md.stride) : int(
                    stop / (md.timestep / 1000) / md.stride
                ) :
            ]
            .mean()
        )
        dataset_std = (
            df["Total MD Energy"]
            .iloc()[
                int(start / (md.timestep / 1000) / md.stride) : int(
                    stop / (md.timestep / 1000) / md.stride
                ) :
            ]
            .std()
        )
        npoints = (
            stop / (md.timestep / 1000) / md.stride
            - start / (md.timestep / 1000) / md.stride
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
                    int(start / (md.timestep / 1000) / md.stride) : int(
                        stop / (md.timestep / 1000) / md.stride
                    )
                ]
                * md.timestep
                / 1000,
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
                .rolling(window=int(10 / (md.timestep / 1000) / md.stride))
                .mean()
                .iloc()[
                    int(start / (md.timestep / 1000) / md.stride) : int(
                        stop / (md.timestep / 1000) / md.stride
                    ) :
                ]
                .min()
                * 23.06
                * (1 + 1e-5),
                df["Total MD Energy"]
                .rolling(window=int(10 / (md.timestep / 1000) / md.stride))
                .mean()
                .iloc()[
                    int(start / (md.timestep / 1000) / md.stride) : int(
                        stop / (md.timestep / 1000) / md.stride
                    ) :
                ]
                .max()
                * 23.06
                * (1 - 1e-5),
            ]
        )

        st.plotly_chart(fig, use_container_width=True)


with convergence_tab:

    # fig2_bins = st.number_input(
    #     label="Number of bins:",
    #     value=100,
    # )

    threshold_energy = st.sidebar.number_input(
        label="Energy convergence threshold (ppm):", value=1
    )

    threshold_time = st.sidebar.number_input(
        label="Time convergence threshold (ps):", value=100
    )

    if st.button("🔃 Calculate convergence data!"):

        for md in selections:

            df = load_output(md)

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            title = md.name

            fig.update_layout(title=title)

            fig.add_trace(
                go.Scatter(
                    x=df.index * (md.timestep / 1000),
                    y=df["Total MD Energy"].expanding().mean() * 23.06,
                    name="Total MD Energy Average",
                ),
                secondary_y=False,
            )

            t_valid = []
            energy_list = df["Total MD Energy"].expanding().mean().to_list()
            for i, _ in enumerate(energy_list):
                exp_av = energy_list[0:i]
                for step, x in enumerate(exp_av[::-1]):
                    dx = abs(exp_av[-1] - x) / abs(exp_av[-1]) * 1e6
                    if dx >= threshold_energy:
                        t_valid.append(step * (md.timestep / 1000) * md.stride)
                        break

            def moving_average(a, n=100):
                ret = np.cumsum(a, dtype=float)
                ret[n:] = ret[n:] - ret[:-n]
                return ret[n - 1 :] / n

            fig.add_trace(
                go.Scatter(
                    x=df.index * (md.timestep / 1000),
                    y=t_valid,
                    name="T valid",
                ),
                secondary_y=True,
            )

            convergence_reached = False
            for i, t in enumerate(t_valid):
                if t > threshold_time:
                    convergence_reached = True
                    converged_step = i
                    break
            if convergence_reached:
                average = (
                    df["Total MD Energy"]
                    .iloc()[: int(converged_step / (md.timestep / 1000) / md.stride) :]
                    .mean()
                )
                dataset_std = (
                    df["Total MD Energy"]
                    .iloc()[: int(converged_step / (md.timestep / 1000) / md.stride) :]
                    .std()
                )
                npoints = converged_step / (md.timestep / 1000) / md.stride
                average_error = dataset_std / np.sqrt((npoints))

                st.write(
                    f"**Average Energy after {converged_step * ( md.timestep / 1000 ) * md.stride} ps:**\n"
                    f"* {average:.{len(f'{average_error:.2}')-2}f} ± {average_error:.2} eV\n"
                    f"* {average*23.06:.{len(f'{average_error*23.06:.2}')-2}f} ± {average_error*23.06:.2} kcal/mol"
                )
            else:
                st.warning(f"Convergence not reached yet for {md.name}!", icon="⚠")

            fig.update_xaxes(title_text="time (ps)")
            fig.update_yaxes(
                title_text="Energy (kcal/mol)",
                range=[
                    df["Total MD Energy"]
                    .expanding()
                    .mean()
                    .iloc()[int(df.index[-1] / 5 / md.stride) : :]
                    .min()
                    * 23.06
                    * (1 + 1e-5),
                    df["Total MD Energy"]
                    .expanding()
                    .mean()
                    .iloc()[int(df.index[-1] / 5 / md.stride) : :]
                    .max()
                    * 23.06
                    * (1 - 1e-5),
                ],
                secondary_y=False,
            )
            fig.update_yaxes(
                title_text="T valid (ps)",
                range=[0, 100],
                secondary_y=True,
            )

            # fig2_start, fig2_stop = st.slider(
            #     label="Calculate energy distribution from/to (ps):",
            #     min_value=int(df.index[0] * ( md.timestep / 1000 )),
            #     max_value=int(df.index[-1] * ( md.timestep / 1000 )),
            #     value=(int(df.index[0] * ( md.timestep / 1000 )), int(df.index[-1] * ( md.timestep / 1000 ))),
            #     key=f"{md.name}_dist",
            # )

            # fig2 = go.Figure(
            #     px.histogram(
            #         x=df["Total MD Energy"].iloc()[
            #             int(fig2_start / ( md.timestep / 1000 ) / md.stride) : int(
            #                 fig2_stop / ( md.timestep / 1000 ) / md.stride
            #             ) :
            #         ],
            #         nbins=int(fig2_bins),
            #     )
            # )
            # fig2.update_layout(title=f"{title} - Distribution of energies")
            # fig2.update_xaxes(title_text="Energy (eV)")
            # fig2.update_yaxes(title_text="Counts")

            st.plotly_chart(fig, use_container_width=True)
            # st.plotly_chart(fig2, use_container_width=True)


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

        df = load_output(md)

        total_weight = 0.0
        try:
            for line in md.mol2:
                if len(line.split()) == 9:
                    total_weight += float(atom_weights[line.split()[1]])
            total_weight *= amu  # g
        except:
            try:
                for line in md.pdb:
                    if len(line.split()) == 11:
                        total_weight += float(atom_weights[line.split()[2]])
                total_weight *= amu  # g
            except:
                st.warning(f"Sorry, no topology file available for {md.name}!")
                continue

        dens_ps = st.slider(
            label="Calculate density at (ps):",
            min_value=int(df.index[0] * (md.timestep / 1000)),
            max_value=int(df.index[-1] * (md.timestep / 1000)),
            value=int(int(df.index[-1] * (md.timestep / 1000))),
            key=f"{md.name}_dens",
        )

        sel_dens = (
            total_weight
            / df["Volume"].iloc()[int(dens_ps / (md.timestep / 1000) / md.stride)]
            / 1e-27
        )
        st.write(f"Density after {dens_ps:.0f} ps: **{sel_dens:.2f}** g/L\n")

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        title = md.name

        fig.update_layout(title=title)

        fig.add_trace(
            go.Scatter(
                x=df.index * (md.timestep / 1000),
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
                x=df.index * (md.timestep / 1000),
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
