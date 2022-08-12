############################################################################################
# # # # # # # # # # # # # # # # #     STREAMLIT PREP     # # # # # # # # # # # # # # # # # #
############################################################################################

from __future__ import annotations
from dataclasses import dataclass
from heapq import merge
from io import BytesIO, TextIOWrapper
from typing import List
import streamlit as st

st.set_page_config(layout="wide",)


@dataclass
class BytesStreamManager:
    name: str
    __stream: BytesIO

    @property
    def bytestream(self) -> BytesIO:
        self.__stream.seek(0)
        return self.__stream

    def __iter__(self):
        text_stream = TextIOWrapper(self.bytestream, encoding="utf-8")
        for line in text_stream:
            yield line
        text_stream.detach()

    def __iadd__(self, obj: BytesStreamManager) -> BytesStreamManager:
        lines: List[str] = []
        for line in self:
            lines.append(line)
        if not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        for line in obj:
            lines.append(line)
        buffer = "".join(lines)
        self.__stream = BytesIO(buffer.encode("utf-8"))
        return self


if "FileBuffer" not in st.session_state:
    st.session_state["FileBuffer"] = []

file_buffer: List[BytesStreamManager] = st.session_state["FileBuffer"]

tab1, tab2, tab3, tab4 = st.tabs(
    ["File Manager", "Energy plot", "Standard Deviation", "Volume/pressure"]
)

with tab1:

    with st.form("File upload form", clear_on_submit=True):
        buffer = st.file_uploader(
            "Select the files to upload", type="out", accept_multiple_files=True
        )
        submitted = st.form_submit_button("Submit")

    if submitted and buffer != [] and buffer is not None:
        for file in buffer:
            file_buffer.append(BytesStreamManager(file.name, BytesIO(file.getvalue())))
        st.experimental_rerun()

    st.session_state["FileBuffer"].sort(key=lambda x: x.name)

    st.markdown("# File list:")
    selections = []
    for file in st.session_state["FileBuffer"]:
        selections.append(st.checkbox(file.name))

    st.markdown("---")

    if st.button("Remove files"):
        newbuffer = []
        for id, file in enumerate(st.session_state["FileBuffer"]):
            if not selections[id]:
                newbuffer.append(file)
        st.session_state["FileBuffer"] = newbuffer
        st.experimental_rerun()

    if st.button("Merge files"):
        newbuffer = []
        merged_files = []

        for id, file in enumerate(st.session_state["FileBuffer"]):
            if selections[id]:
                merged_files.append(file)
            else:
                newbuffer.append(file)

        new_file = merged_files.pop(0)
        while merged_files:
            new_file += merged_files.pop(0)
        newbuffer.append(new_file)
        st.session_state["FileBuffer"] = newbuffer
        st.experimental_rerun()

    newname = st.text_input("Enter new name:")
    if st.button("Rename"):
        for id, file in enumerate(st.session_state["FileBuffer"]):
            if selections[id]:
                file.name = newname
        st.experimental_rerun()

    st.markdown("---")

############################################################################################
# # # # # # # # # # # # # # # # # #     MD FOLLOWER      # # # # # # # # # # # # # # # # # #
############################################################################################

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


@st.cache
def read_md(md_out):

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
            step += mdrestartfreq

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


timestep = float(st.number_input("Timestep (fs): ", value=1.0)) / 1000
mdrestartfreq = int(st.number_input("MD stride: ", value=100))

with tab2:

    # local_options = [item for item in Path(".").glob("**/*md.out")]

    for file in st.session_state["FileBuffer"]:

        df = read_md(file)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        title = file.name

        fig.update_layout(title=title)

        fig.add_trace(
            go.Scatter(
                x=df.index * timestep,
                y=df["Total MD Energy"] * 23.06,
                name="Total MD Energy",
                line={"width": 0.1, "color": "blue",},
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=df.index * timestep,
                y=df["Total MD Energy"]
                .rolling(window=int(10 / timestep / mdrestartfreq))
                .mean()
                * 23.06,
                name="Rolling Average on 10 ps",
                line={"width": 1, "color": "blue",},
            ),
        )

        # fig.add_trace(
        #    go.Scatter(
        #        x=df.index*timestep,
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
                x=df.index[::-1] * timestep,
                y=df["Total MD Energy"].iloc()[::-1].expanding().mean() * 23.06,
                name="Reverse Expanding Average",
                line={"width": 3, "color": "orange",},
            ),
        )
        fig.update_xaxes(title_text="time (ps)")
        fig.update_yaxes(title_text="Energy (kcal/mol)")

        start, stop = st.slider(
            label="Calculate average energy from/to (ps):",
            min_value=int(df.index[0] * timestep),
            max_value=int(df.index[-1] * timestep),
            value=(int(df.index[0] * timestep), int(df.index[-1] * timestep)),
            key=file,
        )

        average = (
            df["Total MD Energy"]
            .iloc()[
                int(start / timestep / mdrestartfreq) : int(
                    stop / timestep / mdrestartfreq
                ) :
            ]
            .mean()
        )
        dataset_std = (
            df["Total MD Energy"]
            .iloc()[
                int(start / timestep / mdrestartfreq) : int(
                    stop / timestep / mdrestartfreq
                ) :
            ]
            .std()
        )
        npoints = stop / timestep / mdrestartfreq - start / timestep / mdrestartfreq
        average_error = dataset_std / np.sqrt((npoints))
        st.write(
            f"Average between {start} and {stop} ps: {average} +/- {average_error} eV, {average*23.06} +/- {average_error*23.06} kcal/mol"
        )

        # fig.add_trace(
        #    go.Scatter(
        #        x=df.index[int(start/timestep/mdrestartfreq):int(stop/timestep/mdrestartfreq)]*timestep,
        #        y=df["Total MD Energy"].iloc()[int(start/timestep/mdrestartfreq):int(stop/timestep/mdrestartfreq):].expanding().mean()*23.06,
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
                    int(start / timestep / mdrestartfreq) : int(
                        stop / timestep / mdrestartfreq
                    )
                ]
                * timestep,
                y=[average * 23.06] * len(df.index),
                name=f"Average energy between {start} and {stop} ps",
                line={"width": 3, "color": "black", "dash": "dash",},
            )
        )
        fig.update_xaxes(range=[start, stop])
        fig.update_yaxes(
            range=[
                df["Total MD Energy"]
                .rolling(window=int(10 / timestep / mdrestartfreq))
                .mean()
                .iloc()[
                    int(start / timestep / mdrestartfreq) : int(
                        stop / timestep / mdrestartfreq
                    ) :
                ]
                .min()
                * 23.06
                * (1 + 1e-5),
                df["Total MD Energy"]
                .rolling(window=int(10 / timestep / mdrestartfreq))
                .mean()
                .iloc()[
                    int(start / timestep / mdrestartfreq) : int(
                        stop / timestep / mdrestartfreq
                    ) :
                ]
                .max()
                * 23.06
                * (1 - 1e-5),
            ]
        )
        st.plotly_chart(fig, use_container_width=True)


with tab3:

    fig2_bins = st.number_input(label="Number of bins:", value=100, key=f"{file}_bins",)

    for file in st.session_state["FileBuffer"]:

        df = read_md(file)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        title = file.name

        fig.update_layout(title=title)

        fig.add_trace(
            go.Scatter(
                x=df.index * timestep,
                y=df["Total MD Energy"].expanding().mean() * 23.06,
                name="Total MD Energy Average",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index * timestep,
                y=df["Total MD Energy"]
                .expanding()
                .mean()
                .rolling(window=int(10 / timestep / mdrestartfreq))
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
                .iloc()[int(df.index[-1] / 5 / mdrestartfreq) : :]
                .min()
                * 23.06
                * (1 + 1e-5),
                df["Total MD Energy"]
                .expanding()
                .mean()
                .iloc()[int(df.index[-1] / 5 / mdrestartfreq) : :]
                .max()
                * 23.06
                * (1 - 1e-5),
            ],
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Standard deviation (kcal/mol)", range=[0, 1], secondary_y=True,
        )

        fig2_start, fig2_stop = st.slider(
            label="Calculate energy distribution from/to (ps):",
            min_value=int(df.index[0] * timestep),
            max_value=int(df.index[-1] * timestep),
            value=(int(df.index[0] * timestep), int(df.index[-1] * timestep)),
            key=file,
        )
        fig2 = go.Figure(
            px.histogram(
                x=df["Total MD Energy"].iloc()[
                    int(fig2_start / timestep / mdrestartfreq) : int(
                        fig2_stop / timestep / mdrestartfreq
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


with tab4:

    for file in st.session_state["FileBuffer"]:

        df = read_md(file)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        title = file.name

        fig.update_layout(title=title)

        fig.add_trace(
            go.Scatter(x=df.index * timestep, y=df["Volume"], name="Volume",),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=df.index * timestep, y=df["Pressure"] / 101325, name="Pressure",),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Volume (A^3)", row=2, secondary_y=False)
        fig.update_yaxes(title_text="Pressure (atm)", row=2, secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

