############################################################################################
# # # # # # # # # # # # # # # # # #     MD ANALYZER      # # # # # # # # # # # # # # # # # #
############################################################################################

import os
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from MDAnalysis.analysis.rdf import InterRDF
from io import StringIO, BytesIO

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tempfile import NamedTemporaryFile as tmp

st.set_page_config(
    layout="wide",
)

ss = st.session_state

options = [ss.MDs[md] for md in ss.MDs]
options.sort(key=lambda x: x.name)
selection = st.sidebar.selectbox(
    "Select MD experiment", options, format_func=lambda x: x.name
)

topo_tmp = tmp(mode="w+")

try:
    if selection.dcd:
        traj_format = "DCD"
        traj_tmp = tmp(mode="wb+")
        traj_tmp.write(BytesIO(selection.dcd.bytestream.getvalue()).read())
except Exception as e:
    # st.write(e)
    try:
        if selection.xyz:
            traj_format = "XYZ"
            traj_tmp = tmp(mode="w+")
            traj_tmp.write(
                StringIO(selection.xyz.bytestream.getvalue().decode("utf-8")).read()
            )
    except Exception as e:
        # st.write(e)
        st.error(f"No trajectory file found for {selection.name}!", icon="❌")
        st.stop()

try:
    topology_format = "PSF"
    topo_tmp.write(StringIO(selection.psf.bytestream.getvalue().decode("utf-8")).read())
    st.success(f".psf file found for {selection.name}!", icon="✔")
    for line in selection.psf:
        if "UNL" in line:
            ss.resname = line.split()[3]
            break
        ss.resname = "UNL"
except Exception as e:
    # st.write(e)
    try:
        topology_format = "PDB"
        topo_tmp.write(StringIO(selection.pdb.bytestream.getvalue().decode("utf-8")).read())
        st.success(f".pdb file found for {selection.name}!", icon="✔")
        for line in selection.pdb:
            if "UNL" in line:
                ss.resname = line.split()[3]
                break
            ss.resname = "UNL"
    except Exception as e:
        # st.write(e)
        try:
            topology_format = "MOL2"
            topo_tmp.write(
                StringIO(selection.mol2.bytestream.getvalue().decode("utf-8")).read()
            )
            st.success(f".mol2 file found for {selection.name}!", icon="✔")
            for line in selection.mol2:
                if "UNL" in line:
                    ss.resname = line.split()[-2]
                    break
                ss.resname = "UNL"
        except Exception as e:
            # st.write(e)
            st.error(f"No topology files available for {selection.name}", icon="❌")
            st.stop()


def create_u():

    u = mda.Universe(
        topo_tmp.name, traj_tmp.name, format=traj_format, topology_format=topology_format
    )
    try:
        ss.box_side = float(selection.pbc.bytestream.read())
        u.dimensions = [ss.box_side, ss.box_side, ss.box_side, 90, 90, 90]
    except:
        try:
            ss.box_side = u.dimensions[0]
        except:
            st.error(f"No PBC info found!", icon="❌")
            st.stop()
    return u


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "💫 Atom-Atom RDF",  # 1
        "🌊 Linear Density",  # 2
        "🏃‍♀️ Self-Diffusivity",  # 3
        "⚡ Dielectric Constant",  # 4
        "💠 Solute-solvent RDF",  # 5
        "📈 2D RDF",  # 6
    ]
)

with tab1:

    # Atom-Atom Radial Distribution Function (RDF)

    tab1_col1, tab1_col2 = st.columns(2)

    with tab1_col1:

        show_water = st.checkbox("System contains water")
        show_AN = st.checkbox("System contains acetonitrile")

        atom1 = st.text_input("Select atom type 1: ", value="O")
        atom2 = st.text_input("Select atom type 2: ", value="O")

    u = create_u()

    solvent = u.select_atoms(f"not resname {ss.resname}")

    workflow = [
        trans.unwrap(u.atoms),
        trans.wrap(u.atoms, compound="residues"),
    ]
    u.trajectory.add_transformations(*workflow)

    rdf = InterRDF(
        u.select_atoms(f"name {atom1}"),
        u.select_atoms(f"name {atom2}"),
        nbins=500,
        range=([0.0, ss.box_side / 2]),
        exclusion_block=(1, 1),
    )

    with tab1_col2:

        if "fig_rdf" not in ss or st.button(f"🧹 Clear {atom1}-{atom2} RDF"):
            ss.fig_rdf = go.Figure()

            import os

            if show_water:
                exp_path = (
                    f"{os.path.dirname(__file__)}/../data/RDF_water_{atom1}{atom2}.csv"
                )
                exp = pd.read_csv(exp_path)
                ss.fig_rdf.add_trace(
                    go.Scatter(
                        x=exp["r (Å)"],
                        y=exp[f"g_{atom1}{atom2}"],
                        name="Experimental",
                        line={
                            "color": "black",
                            "dash": "dash",
                        },
                    ),
                )
            if show_AN:
                exp_path = f"{os.path.dirname(__file__)}/../data/RDF_acetonitrile_{atom1}{atom2}.csv"
                exp = pd.read_csv(exp_path)
                ss.fig_rdf.add_trace(
                    go.Scatter(
                        x=exp["r (Å)"],
                        y=exp[f"g_{atom1}{atom2}"],
                        name="Experimental",
                        line={
                            "color": "black",
                            "dash": "dash",
                        },
                    ),
                )

        if "rdf_atom" not in ss:
            ss["rdf_atom"] = None
        if st.button(f"🔃 Calculate {atom1}-{atom2} RDF"):
            ss["rdf_atom"] = rdf.run(step=1)
            rdf_atom = ss["rdf_atom"]

            ss.fig_rdf.add_trace(
                go.Scatter(
                    x=rdf_atom.results.bins,
                    y=rdf_atom.results.rdf,
                    name=selection.name,
                ),
            )

            ss.fig_rdf.update_xaxes(title_text="r (Å)")
            ss.fig_rdf.update_yaxes(title_text=f"g(r) {atom1}-{atom2}")

    st.plotly_chart(ss.fig_rdf, use_container_width=True)

with tab2:

    # Linear Density

    u = create_u()

    from MDAnalysis.analysis.lineardensity import LinearDensity

    ldens = LinearDensity(u.atoms, binsize=0.1)

    if "ldens" not in ss:
        ss["ldens"] = None
    if st.button("🔃 Calculate Linear Density"):
        ss["ldens"] = ldens.run()
        ldens = ss["ldens"]

        ss.fig_ldens = go.Figure()
        average = (
            ldens.results.x.mass_density
            + ldens.results.y.mass_density
            + ldens.results.z.mass_density
        ) / 3

        ss.fig_ldens.add_trace(
            go.Scatter(
                x=ldens.results.x.hist_bin_edges,
                y=ldens.results.x.mass_density,
                name="X",
                line={
                    "width": 0.5,
                    "color": "red",
                },
            ),
        )
        ss.fig_ldens.add_trace(
            go.Scatter(
                x=ldens.results.y.hist_bin_edges,
                y=ldens.results.y.mass_density,
                name="Y",
                line={
                    "width": 0.5,
                    "color": "green",
                },
            ),
        )
        ss.fig_ldens.add_trace(
            go.Scatter(
                x=ldens.results.z.hist_bin_edges,
                y=ldens.results.z.mass_density,
                name="Z",
                line={
                    "width": 0.5,
                    "color": "blue",
                },
            ),
        )
        ss.fig_ldens.add_trace(
            go.Scatter(
                x=ldens.results.z.hist_bin_edges,
                y=average,
                name="Average",
                line={
                    "width": 3,
                    "color": "black",
                },
            ),
        )

        ss.fig_ldens.update_xaxes(title_text="position (Å)")
        ss.fig_ldens.update_yaxes(title_text="density (kg/L)")
        st.plotly_chart(ss.fig_ldens, use_container_width=True)

with tab3:

    # Mean Squared Displacement (MSD)

    u = create_u()

    import MDAnalysis.analysis.msd as msd

    MSD = msd.EinsteinMSD(u, select="all", msd_type="xyz", fft=True)

    if "MSD" not in ss:
        ss["MSD"] = MSD.run()
    if st.button("🔃 Calculate MSD"):
        ss["MSD"] = MSD.run()
    MSD = ss["MSD"]

    msd = MSD.results.timeseries

    nframes = MSD.n_frames
    timestep = 100  # this needs to be the actual time between frames
    st.write(f"Calculating MSD with a timestep of {timestep} fs")
    lagtimes = np.arange(nframes) * timestep  # make the lag-time axis

    fig_msd = make_subplots(specs=[[{"secondary_y": True}]])

    fig_msd.add_trace(
        go.Scatter(
            x=lagtimes,
            y=msd,
            name="MSD",
        ),
    )

    # Calculating self-diffusivity

    from scipy.stats import linregress

    start_time, end_time = st.slider(
        label="Select start and end time (ps):",
        min_value=int(lagtimes[0]),
        max_value=int(lagtimes[-1]),
        value=(int(lagtimes[0]), int(lagtimes[-1])),
    )
    start_index = int(start_time / timestep)
    end_index = int(end_time / timestep)

    fig_msd.add_trace(
        go.Scatter(
            x=np.arange(start_time, end_time),
            y=np.arange(start_time, end_time),
            name="slope = 1",
            line={
                "dash": "dash",
            },
        ),
        secondary_y=True,
    )
    fig_msd.update_xaxes(
        range=[start_time, end_time],
        title_text="lagtime (fs)",
        # type="log",
    )
    fig_msd.update_yaxes(
        range=[msd[start_time // timestep], msd[end_time // timestep]],
        title_text="MSD (Å^2 / fs)",
        # type="log",
    )
    fig_msd.update_yaxes(
        title_text="",
        range=[start_time, end_time],
        secondary_y=True,
        # type="log",
    )

    linear_model = linregress(lagtimes[start_index:end_index], msd[start_index:end_index])
    slope = linear_model.slope
    error = linear_model.rvalue
    # dim_fac is 3 as we computed a 3D msd with 'xyz'
    D = slope * 1 / (2 * MSD.dim_fac)
    st.write(f"Self-diffusivity coefficient: {(D*(10**-5)):.3E} m\N{SUPERSCRIPT TWO}/s")

    st.plotly_chart(fig_msd, use_container_width=True)

with tab4:

    # Dielectric constant

    u = create_u()

    from MDAnalysis.analysis.dielectric import DielectricConstant

    diel = DielectricConstant(u.atoms, temperature=298.15, make_whole=True)

    if "diel" not in ss:
        ss["diel"] = None
    if st.button("🔃 Calculate Dielectric Constant"):
        ss["diel"] = diel.run()
        diel = ss["diel"]
        st.write(f"Dielectric constant: {diel.results.eps_mean:.3}")

with tab5:

    # Solute-solvent Radial Distribution Function (RDF)

    u = create_u()

    solute = u.select_atoms(f"resname {ss.resname}")
    solvent = u.select_atoms(f"not resname {ss.resname}")

    workflow = [
        trans.unwrap(u.atoms),
        trans.wrap(u.atoms, compound="residues"),
    ]
    u.trajectory.add_transformations(*workflow)

    rdf = InterRDF(solute, solvent, nbins=500, range=[1.0, ss.box_side / 2])

    if "fig_solvrdf" not in ss or st.button("🧹 Clear Solute-solvent RDF plot"):
        ss.fig_solvrdf = go.Figure()

    if "rdf_solute_solvent" not in ss:
        ss["rdf_solute_solvent"] = None
    if st.button("🔃 Calculate solute-solvent RDF"):
        ss["rdf_solute_solvent"] = rdf.run(step=1)
        rdf_solute_solvent = ss["rdf_solute_solvent"]

        ss.fig_solvrdf.add_trace(
            go.Scatter(
                x=rdf_solute_solvent.results.bins,
                y=rdf_solute_solvent.results.rdf,
                name="Solute-solvent RDF",
            ),
        )

        ss.fig_solvrdf.update_xaxes(title_text="r (Å)")
        ss.fig_solvrdf.update_yaxes(title_text="g(r)")

    st.plotly_chart(ss.fig_solvrdf, use_container_width=True)


with tab6:

    # 2-dimensional RDF (vs time)

    atom1 = st.text_input("Select atom type 1 : ", value="O")
    atom2 = st.text_input("Select atom type 2 : ", value="O")

    u = create_u()

    solvent = u.select_atoms(f"not resname {ss.resname}")

    workflow = [
        trans.unwrap(u.atoms),
        trans.wrap(u.atoms, compound="residues"),
    ]
    u.trajectory.add_transformations(*workflow)

    rdf_2D = InterRDF(
        u.select_atoms(f"name {atom1}"),
        u.select_atoms(f"name {atom2}"),
        nbins=500,
        range=([0.0, ss.box_side / 2]),
        exclusion_block=(1, 1),
    )

    stride = st.number_input(
        "Stride (frames): ",
        min_value=1,
        max_value=len(u.trajectory),
        value=50,
    )
    zmin = st.number_input("z min: ", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
    zmax = st.number_input("z max: ", min_value=0.0, max_value=10.0, value=1.5, step=0.1)

    if "fig_2Drdf" not in ss:
        ss.fig_2Drdf = None
    if st.button(f"🔃 Calculate {atom1}-{atom2} 2D RDF"):
        ss.fig_2Drdf = go.Figure()

        step = 0
        progress = st.progress(0)
        rdf_2D_surface = []
        times = []

        for i in range(len(u.trajectory) // stride):

            rdf_2D.run(start=step, stop=step + stride, step=1)

            rdf_2D_surface.append(rdf_2D.results.rdf)
            times.append(step)
            step += stride

            progress.progress((i + 1) / (len(u.trajectory) // stride))

        ss.fig_2Drdf.add_trace(
            go.Contour(
                x=rdf_2D.results.bins,
                z=rdf_2D_surface,
                y=times,
                zmin=zmin,
                zmax=zmax,
                name=selection.name,
                contours_coloring="heatmap",
                colorbar=dict(title=f"g(r) {atom1}-{atom2}"),
                line_width=0,
            ),
        )

        ss.fig_2Drdf.update_xaxes(title_text="r (Å)")
        ss.fig_2Drdf.update_yaxes(title_text="frame #")

        st.plotly_chart(ss.fig_2Drdf, use_container_width=True)

topo_tmp.close()
traj_tmp.close()
