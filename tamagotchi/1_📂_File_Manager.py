############################################################################################
# # # # # # # # # # # # # # # # #     STREAMLIT PREP     # # # # # # # # # # # # # # # # # #
############################################################################################

from __future__ import annotations  # do we need this?
from dataclasses import dataclass
from io import BytesIO, TextIOWrapper
from typing import List
import streamlit as st
import os
from time import sleep

st.set_page_config(layout="wide")
ss = st.session_state


class MD:
    def __init__(self, name):
        self.name = name

    # def __getattr__(self, attr):
    #     try:
    #         return self.__getattribute__(attr)
    #     except AttributeError:
    #         st.warning(f"Attribute {attr} not found!")
    #         return None


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


ss.timestep = float(st.sidebar.number_input("Timestep (fs): ", value=1.0)) / 1000
ss.mdrestartfreq = int(st.sidebar.number_input("MD stride: ", value=100))

with upload_tab:

    with st.form("File upload form", clear_on_submit=True):
        buffer = st.file_uploader(
            "Select the files to upload",
            type=["xyz", "mol2", "pdb", "psf", "pbc", "out", "namd", "dcd"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("📤 Submit")

    # --> finding local files in scratch directory
    import glob

    local_files = []
    local_files += glob.glob("/scratch/lbabetto/**/*md.out", recursive=True)
    local_files += glob.glob("/scratch/lbabetto/**/*geo_end.xyz", recursive=True)
    local_files += glob.glob("/scratch/lbabetto/**/*.mol2", recursive=True)
    local_files += glob.glob("/scratch/lbabetto/**/*.pdb", recursive=True)
    local_files += glob.glob("/scratch/lbabetto/**/*.pbc", recursive=True)
    local_files += glob.glob("/scratch/lbabetto/**/*.psf", recursive=True)
    local_files += glob.glob("/scratch/lbabetto/**/*.dcd", recursive=True)

    for file in local_files:
        if file not in [file.name for file in ss.FileBuffer]:
            with open(file, "rb") as f:
                ss.FileBuffer.append(BytesStreamManager(file, BytesIO(f.read())))
    # <---

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
                if os.path.splitext(file_1.name)[0] not in ss.MDs and os.path.splitext(
                    file_1.name
                )[1] in [".xyz", ".out", ".namd", ".dcd"]:

                    basename_1 = (
                        os.path.splitext(os.path.basename(file_1.name))[0]
                        .replace("_geo_end", "")
                        .replace("_md", "")
                    )

                    # rename "single runs" not done via GES-comp-echem
                    if basename_1 == "geo_end" or basename_1 == "md":
                        basename_1 = os.path.basename(os.path.dirname(file_1.name))

                    md = MD(basename_1)

                    for file_2 in ss.FileBuffer:
                        basename_2 = (
                            os.path.splitext(os.path.basename(file_2.name))[0]
                            .replace("_geo_end", "")
                            .replace("_md", "")
                        )
                        # rename "single runs" not done via GES-comp-echem
                        if basename_2 == "geo_end" or basename_2 == "md":
                            basename_2 = os.path.basename(os.path.dirname(file_2.name))

                        if basename_2 == md.name:
                            setattr(md, os.path.splitext(file_2.name)[1][1:], file_2)

                    ss.MDs[md.name] = md

        st.write("---")
        for md in ss.MDs:
            st.write(f"##### {ss.MDs[md].name}")
            for property in dir(ss.MDs[md]):
                if not property.startswith("__") and property != "name":
                    st.write(f"* {property}: ``{getattr(ss.MDs[md], property).name}``")
            st.write("---")

    with setup_col2:

        st.write("### Edit MD:")
        md_selections = st.multiselect(
            "Select MD simulations to edit:", ss.MDs, format_func=lambda x: ss.MDs[x].name
        )

        st.write("Append trajectory data:")
        append_file = st.selectbox(
            "Select trajectory file to append",
            [file for file in ss.FileBuffer if os.path.splitext(file.name)[1] == ".xyz"],
            format_func=lambda x: x.name,
        )

        if st.button("🔗 Add trajectory file"):
            if len(md_selections) > 1:
                st.error("You should append trajectory files to only one file at a time!")
            else:
                ss.MDs[md_selections[0]].xyz += append_file
                st.success(
                    f"{append_file.name} added to {ss.MDs[md_selections[0]].name}", icon="✅"
                )
                sleep(1)
                st.experimental_rerun()

        rename_string = st.text_input("Rename MD:")
        if st.button("📝 Rename MD"):
            for md_selection in md_selections:
                ss.MDs[md_selection].name = rename_string
                st.success(
                    f"{ss.MDs[md_selection].name} renamed to {rename_string}", icon="✅"
                )
                sleep(1)
            st.experimental_rerun()

        if st.button("❌ Remove MD"):
            for md_selection in md_selections:
                del ss.MDs[md_selection]
                st.success(f"{md_selection} removed", icon="✅")
                sleep(1)
            st.experimental_rerun()

        st.write("Overwrite MD data:")

        extensions = []
        for file in ss.FileBuffer:
            ext = os.path.splitext(file.name)[1]
            if ext not in extensions:
                extensions.append(ext)

        file_type = st.selectbox("File type:", extensions)
        overwrite_file = st.selectbox(
            "Select file with new data",
            [file for file in ss.FileBuffer if os.path.splitext(file.name)[1] == file_type],
            format_func=lambda x: x.name,
        )
        if st.button("💿 Overwrite MD data"):
            for md_selection in md_selections:
                setattr(ss.MDs[md_selection], file_type, overwrite_file)
                st.success(
                    f"{md_selection} .{file_type} set to {overwrite_file.name}", icon="✅"
                )
                sleep(1)
            st.experimental_rerun()
