############################################################################################
# # # # # # # # # # # # # # # # #     STREAMLIT PREP     # # # # # # # # # # # # # # # # # #
############################################################################################

from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO, TextIOWrapper
from typing import List
import streamlit as st
import os

st.set_page_config(layout="wide")
ss = st.session_state


@dataclass
class BytesStreamManager:
    name: str
    __stream: BytesIO

    @property
    def filename(self):
        return self.name

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


if "FileBuffer" not in ss:
    ss.FileBuffer = []

tab1, tab2, tab3 = st.tabs(
    [
        "Upload",
        "Edit",
        "Setup",
    ]
)


def merge_files(selections, filebuffer):

    newbuffer = []
    merged_files = []

    for id, file in enumerate(ss[filebuffer]):
        if selections[id]:
            merged_files.append(file)
        else:
            newbuffer.append(file)

    new_file = merged_files.pop(0)
    while merged_files:
        new_file += merged_files.pop(0)
    newbuffer.append(new_file)
    ss[filebuffer] = newbuffer
    st.experimental_rerun()


def remove_files(selections, filebuffer):
    newbuffer = []
    for id, file in enumerate(ss[filebuffer]):
        if not selections[id]:
            newbuffer.append(file)
    ss[filebuffer] = newbuffer
    st.experimental_rerun()


def rename_files(selections, filebuffer, newname):
    for id, file in enumerate(ss[filebuffer]):
        if selections[id]:
            file.name = newname
    st.experimental_rerun()


with tab1:

    with st.form("File upload form", clear_on_submit=True):
        buffer = st.file_uploader(
            "Select the files to upload",
            type=["xyz", "mol2", "pbc", "out"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Submit")

    if submitted and buffer != [] and buffer is not None:
        for file in buffer:
            if os.path.splitext(file.name)[1] == ".out":
                ss.FileBuffer.append(
                    BytesStreamManager(file.name, BytesIO(file.getvalue()))
                )
            # if os.path.splitext(file.name)[1] == ".xyz":
            #     ss.XYZFileBuffer.append(
            #         BytesStreamManager(file.name, BytesIO(file.getvalue()))
            #     )
            # elif os.path.splitext(file.name)[1] == ".mol2":
            #     ss.MOL2FileBuffer.append(
            #         BytesStreamManager(file.name, BytesIO(file.getvalue()))
            #     )
            # elif os.path.splitext(file.name)[1] == ".pbc":
            #     ss.PBCFileBuffer.append(
            #         BytesStreamManager(file.name, BytesIO(file.getvalue()))
            #     )
        st.experimental_rerun()

    ss.FileBuffer.sort(key=lambda x: x.name)

    if ss.FileBuffer != []:
        st.write("**Output files:**")
        selections = []
        for file in ss.FileBuffer:
            selections.append(st.checkbox(file.name, key=file))
        if st.button("Remove output files"):
            remove_files(selections, "FileBuffer")

with tab2:

    st.write("**Output files:**")
    selection = st.selectbox(
        "Select output file", [traj.name for traj in ss.FileBuffer]
    )
    st.write("---")
    out_rename = st.text_input("New output name: ")
    if st.button("Rename output file"):
        rename_files(out_rename, "FileBuffer", out_rename)

with tab3:
    ss.timestep = float(st.number_input("Timestep (fs): ", value=1.0)) / 1000
    ss.mdrestartfreq = int(st.number_input("MD stride: ", value=100))
