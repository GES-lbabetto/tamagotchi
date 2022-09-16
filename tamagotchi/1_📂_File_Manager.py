############################################################################################
# # # # # # # # # # # # # # # # #     STREAMLIT PREP     # # # # # # # # # # # # # # # # # #
############################################################################################

from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO, StringIO, TextIOWrapper
from typing import List
import streamlit as st
import os

st.set_page_config(layout="wide")
ss = st.session_state


class MD:
    def __init__(self, name: str):
        self.name = name
        self.out = None
        self.xyz = None
        self.mol2 = None
        self.pbc = None


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


# Initializing file buffers
if "FileBuffer" not in ss:
    ss.FileBuffer = []
if "MDs" not in ss:
    ss.MDs = {}

upload_tab, setup_tab = st.tabs(
    [
        "📤 Upload",
        "✍ Setup",
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


ss.timestep = float(st.sidebar.number_input("Timestep (fs): ", value=1.0)) / 1000
ss.mdrestartfreq = int(st.sidebar.number_input("MD stride: ", value=100))

with upload_tab:

    with st.form("File upload form", clear_on_submit=True):
        buffer = st.file_uploader(
            "Select the files to upload",
            type=["xyz", "mol2", "pbc", "out"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("📤 Submit")

    import glob

    local_files = []
    local_files += glob.glob("/scratch/lbabetto/**/*md.out", recursive=True)
    local_files += glob.glob("/scratch/lbabetto/**/*geo_end.xyz", recursive=True)
    local_files += glob.glob("/scratch/lbabetto/**/*.mol2", recursive=True)
    local_files += glob.glob("/scratch/lbabetto/**/*.pbc", recursive=True)

    for file in local_files:
        if file not in [file.name for file in ss.FileBuffer]:
            with open(file, "rb") as f:
                ss.FileBuffer.append(BytesStreamManager(file, BytesIO(f.read())))

    if submitted and buffer != [] and buffer is not None:
        for file in buffer:
            ss.FileBuffer.append(BytesStreamManager(file.name, BytesIO(file.getvalue())))
        st.experimental_rerun()

    ss.FileBuffer.sort(key=lambda x: x.name)
    if ss.FileBuffer != []:
        st.write("### Loaded files:")
        for file in ss.FileBuffer:
            st.write(f"* {file.name}")

with setup_tab:

    setup_col1, setup_col2 = st.columns(2)

    with setup_col1:

        st.write("### Available MDs:")
        if st.button("🔃 Initialize MDs"):
            ss.MDs = {}
            for file_1 in ss.FileBuffer:
                if os.path.splitext(file_1.name)[0] not in ss.MDs:

                    basename_1 = (
                        os.path.splitext(os.path.basename(file_1.name))[0]
                        .replace("_geo_end", "")
                        .replace("_md", "")
                    )

                    md = MD(basename_1)

                    for file_2 in ss.FileBuffer:
                        basename_2 = (
                            os.path.splitext(os.path.basename(file_2.name))[0]
                            .replace("_geo_end", "")
                            .replace("_md", "")
                        )
                        if basename_2 == md.name:
                            setattr(md, os.path.splitext(file_2.name)[1][1:], file_2)

                    ss.MDs[md.name] = md

        st.write("---")
        for md in ss.MDs:
            st.write(f"##### {ss.MDs[md].name}")
            st.write(f"* output: ``{ss.MDs[md].out.name if ss.MDs[md].out else None}``")
            st.write(f"* trajectory: ``{ss.MDs[md].xyz.name if ss.MDs[md].xyz else None}``")
            st.write(f"* topology: ``{ss.MDs[md].mol2.name if ss.MDs[md].mol2 else None}``")
            st.write(f"* pbc: ``{ss.MDs[md].pbc.name if ss.MDs[md].pbc else None}``")
            st.write("---")

    with setup_col2:

        st.write("### Edit MD:")
        md_selections = st.multiselect(
            "Select MD:", ss.MDs, format_func=lambda x: ss.MDs[x].name
        )

        rename_string = st.text_input("Rename MD:")
        if st.button("📝 Rename MD"):
            for md_selection in md_selections:
                ss.MDs[md_selection].name = rename_string
            st.experimental_rerun()
        if st.button("❌ Remove MD"):
            for md_selection in md_selections:
                del ss.MDs[md_selection]
            st.experimental_rerun()

        st.write("Overwrite MD data:")
        file_type = st.selectbox("File type:", ["out", "xyz", "mol2", "pbc"])
        overwrite_file = st.selectbox(
            "Select file with new data",
            [
                file
                for file in ss.FileBuffer
                if os.path.splitext(file.name)[1] == f".{file_type}"
            ],
            format_func=lambda x: x.name,
        )
        if st.button("💿 Overwrite MD data"):
            for md_selection in md_selections:
                setattr(ss.MDs[md_selection], file_type, overwrite_file)
            st.experimental_rerun()
