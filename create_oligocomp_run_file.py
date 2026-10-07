import json
import re
import shlex
import time

import streamlit as st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_name(value: str, fallback: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9\s\-_]", "", value or "")
    value = value.replace(" ", "_").strip("_")
    return value or fallback


def python_list_string(value: str) -> str:
    """Convert comma-separated atom names to the Python-list syntax expected by OligoComp."""
    atoms = [item.strip() for item in value.split(",") if item.strip()]
    return repr(atoms)


def bash_double_quote(value) -> str:
    """Safely place a value inside a shell double-quoted string."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")


def add_optional_arg(parts, flag, value):
    if value is not None and value != "":
        parts.extend([flag, shlex.quote(str(value))])


def build_command(calc, glob):
    """
    Build one `python3 -m oligocomp ...` command.

    NOTE: OligoComp currently exposes the atom-wise switch as `--atomewise`
    (spelling preserved deliberately to match the existing CLI).
    """
    parts = [
        "python3", "-m", "oligocomp",
        '"$PSF_FILE"', "$DCD_FILES",
        "--output_folder", f'"{calc["output_var"]}"',
        "--max_water", str(calc["max_water"]),
        "--occupancy", str(calc["occupancy"]),
        "--conservation_threshold", str(calc["conservation_threshold"]),
        "--distance", str(calc["distance"]),
        "--cut_angle", str(calc["cut_angle"]),
        "--selection", f'"{bash_double_quote(calc["selection"])}"',
        "--additional_donors", f'"{bash_double_quote(glob["donors"])}"',
        "--additional_acceptors", f'"{bash_double_quote(glob["acceptors"])}"',
        "--step", str(calc["step"]),
        "--wrap_dcd", str(calc["wrap_dcd"]).lower(),
        "--plot_parameters", '"$PLOT_PARAMETERS"',
        "--res_id_label_shift", str(calc["res_id_label_shift"]),
    ]

    if calc["segment_names"]:
        parts.append("--segment_names")
        parts.extend(shlex.quote(seg) for seg in calc["segment_names"])

    if calc["start"] is not None:
        parts.extend(["--start", str(calc["start"])])
    if calc["stop"] is not None:
        parts.extend(["--stop", str(calc["stop"])])

    if not calc["residuewise"]:
        parts.append("--atomewise")

    if calc["include_backbone"]:
        parts.append("--include_backbone")
    if calc["no_label_plots"]:
        parts.append("--no_label_plots")
    if calc["dont_save_graph_objects"]:
        parts.append("--dont_save_graph_objects")
    if calc["collect_angles"]:
        parts.append("--collect_angles")

    mode = calc["mode"]
    if mode == "Connected component from root":
        parts.extend(["--root", f'"{bash_double_quote(calc["root_node"])}"'])
    elif mode == "Path between two nodes":
        parts.extend([
            "--path",
            f'"{bash_double_quote(calc["start_node"])}"',
            f'"{bash_double_quote(calc["goal_node"])}"',
        ])

    # Make a readable multiline command.
    continuation_indent = " " * 8
    lines = []
    current = []
    for token in parts:
        current.append(token)
        # Break mostly after option/value pairs to keep preview readable.
        if len(" ".join(current)) > 92 and len(current) > 2:
            lines.append(" ".join(current))
            current = []
    if current:
        lines.append(" ".join(current))

    command = (" \\\n" + continuation_indent).join(lines)
    command += ' >> "$LOGFILE" 2>&1'
    return command


def generate_bash(env, glob, calculations):
    bash = f"""#!/bin/bash
# ================================================================
# OligoComp run file generated with the Streamlit run-file generator
# ================================================================

set -u

if command -v module &>/dev/null; then
    echo "Module system detected. Loading Python..."
    module load Python
fi

source {shlex.quote(env["venv_path"])}

# --- Logging -----------------------------------------------------
OUTPUT_FOLDER={shlex.quote(env["output_folder"])}
mkdir -p "$OUTPUT_FOLDER"
LOGFILE="$OUTPUT_FOLDER/oligocomp_{env["run_name"]}.log"

echo "===============================================================" >> "$LOGFILE"
echo "$(date '+%Y-%m-%d %H:%M:%S'): Starting OligoComp workflow..." >> "$LOGFILE"
echo "Run-file name: run_{env["run_name"]}" >> "$LOGFILE"

# --- Input files -------------------------------------------------
PSF_FILE={shlex.quote(env["psf"])}
DCD_FILES=$(ls {env["dcd"]})

echo "PSF: $PSF_FILE" >> "$LOGFILE"
echo "DCD files: $DCD_FILES" >> "$LOGFILE"

cd {shlex.quote(env["oligocomp_dir"])} || exit 1

# --- Shared parameters ------------------------------------------
PLOT_PARAMETERS={shlex.quote(glob["plot_parameters"])}

"""

    for i, calc in enumerate(calculations, start=1):
        safe_name = clean_name(calc["name"], f"calculation_{i}")
        calc_var = f"CALC_{i}_FOLDER"
        calc["output_var"] = f"${{{calc_var}}}"

        bash += f"""# ================================================================
# Calculation {i}: {safe_name}
# ================================================================
{calc_var}="$OUTPUT_FOLDER/{safe_name}"
mkdir -p "${{{calc_var}}}"

echo "$(date '+%Y-%m-%d %H:%M:%S'): Starting calculation {i}: {safe_name}" >> "$LOGFILE"
"""

        cmd = build_command(calc, glob)

        if calc["run_in_background"]:
            cmd += " &"

        bash += cmd + "\n\n"

        if calc["run_in_background"]:
            bash += "# This calculation runs in the background.\n\n"

    if any(c["run_in_background"] for c in calculations):
        bash += 'echo "$(date \'+%Y-%m-%d %H:%M:%S\'): Waiting for background calculations..." >> "$LOGFILE"\n'
        bash += "wait\n\n"

    bash += """echo "$(date '+%Y-%m-%d %H:%M:%S'): Finished OligoComp workflow." >> "$LOGFILE"
deactivate
"""
    return bash


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="OligoComp Run File Generator", layout="wide")
st.title("OligoComp Run File Generator")
st.caption(
    "Create a bash run file for one or more OligoComp calculations. "
    "The generated command-line arguments follow the current oligocomp.py CLI."
)

st.header("Environment parameters")
st.info(
    "These paths are interpreted from the location where the generated run file is executed. "
    "Absolute paths are recommended for PSF, DCD and output data."
)

c1, c2 = st.columns(2)
with c1:
    venv_path = st.text_input(
        "Virtual environment activation script",
        value="./oligo_comp/bin/activate",
        help="Example: ~/.venvs/oligo_comp/bin/activate",
    ).strip()

with c2:
    oligocomp_dir = st.text_input(
        "OligoComp installation directory",
        value="./OligoComp",
        help="The run file changes into this directory before executing `python3 -m oligocomp`.",
    ).strip()

st.divider()
st.header("Global input files and output")

output_folder = st.text_input(
    "Output folder - required",
    help="Top-level folder for the log file and generated calculation folders.",
).strip()

run_name = clean_name(
    st.text_input(
        "Name of the run",
        value="oligocomp_calculation",
        help="Used in the run-file name and logfile name.",
    ),
    "oligocomp_calculation",
)

psf = st.text_input(
    "PSF file - required",
    help="Path to the PSF/topology file.",
).strip()

dcd = st.text_input(
    "DCD file(s) - required",
    help=(
        "Shell expression used inside `ls`, so wildcards and pipes are possible. "
        "Example: /data/traj_*.dcd | grep -v 'PBC.dcd'"
    ),
).strip()

if output_folder:
    st.caption(f"Log file: `{output_folder}/oligocomp_{run_name}.log`")

if not output_folder or not psf or not dcd:
    st.warning(
        "Fill in Output folder, PSF file and DCD file(s) to configure the calculations."
    )
    st.stop()

st.divider()
st.header("Shared parameters")

c1, c2 = st.columns(2)
with c1:
    donor_text = st.text_input(
        "Additional donors",
        value="",
        help="Comma-separated atom names, e.g. `N, S`.",
    )
with c2:
    acceptor_text = st.text_input(
        "Additional acceptors",
        value="",
        help="Comma-separated atom names, e.g. `O, F`.",
    )

donors = python_list_string(donor_text)
acceptors = python_list_string(acceptor_text)

with st.expander("Adjust graph plot visualization", expanded=False):
    p1, p2 = st.columns(2)
    with p1:
        edge_width = st.number_input("Edge width", min_value=1, value=2, step=1)
        node_size = st.number_input("Node size", min_value=1, value=150, step=1)
        node_label_size = st.number_input("Node label size", min_value=1, value=12, step=1)
        edge_label_size = st.number_input("Edge label size", min_value=1, value=10, step=1)

        graph_color = st.color_picker("Graph color", "#808080")
        water_node_color = st.color_picker("Water node color", "#db5c5c")
        difference_graph_color = st.color_picker("Difference graph color", "#129fe6")
        non_prot_color = st.color_picker("Non-protein node color", "#008000")

    with p2:
        plot_title_fontsize = st.number_input("Title font size", min_value=1, value=20)
        plot_label_fontsize = st.number_input("Axis-label font size", min_value=1, value=36)
        plot_tick_fontsize = st.number_input("Tick font size", min_value=1, value=33)
        plot_resolution = st.number_input("Resolution (DPI)", min_value=72, value=400)

        fig_width = st.number_input("Figure width", min_value=1, value=15)
        fig_height = st.number_input("Figure height", min_value=1, value=16)
        formats = st.multiselect(
            "Export formats",
            ["png", "pdf", "svg", "eps"],
            default=["png"],
        )
        show_chain_label = st.checkbox("Show chain label", value=False)

plot_parameters = {
    "edge_width": edge_width,
    "node_label_size": node_label_size,
    "edge_label_size": edge_label_size,
    "node_size": node_size,
    "graph_color": graph_color,
    "water_node_color": water_node_color,
    "difference_graph_color": difference_graph_color,
    "non_prot_color": non_prot_color,
    "plot_title_fontsize": plot_title_fontsize,
    "plot_label_fontsize": plot_label_fontsize,
    "plot_tick_fontsize": plot_tick_fontsize,
    "plot_resolution": plot_resolution,
    "figsize": (fig_width, fig_height),
    "formats": formats or ["png"],
    "show_chain_label": show_chain_label,
}

st.info(
    "Current OligoComp note: `--inter_monomer` cannot be switched off through the CLI "
    "because the argument is defined with `action='store_true'` and `default=True`. "
    "Therefore this generator does not add a misleading on/off control for it."
)

st.divider()
st.header("OligoComp calculations")

if "oligocomp_calculations" not in st.session_state:
    st.session_state.oligocomp_calculations = [
        {
            "uid": str(time.time()),
            "name": "full_graph",
        }
    ]


def add_calculation():
    if len(st.session_state.oligocomp_calculations) >= 8:
        st.warning("Maximum 8 calculations are supported in one generated run file.")
        return
    st.session_state.oligocomp_calculations.append(
        {
            "uid": str(time.time()),
            "name": f"calculation_{len(st.session_state.oligocomp_calculations) + 1}",
        }
    )


top1, top2 = st.columns([1, 1])
if top1.button("➕ Add calculation"):
    add_calculation()
    st.rerun()

if top2.button("Reset calculations"):
    st.session_state.oligocomp_calculations = [
        {"uid": str(time.time()), "name": "full_graph"}
    ]
    st.rerun()

calculations = []

for i, item in enumerate(st.session_state.oligocomp_calculations):
    uid = item["uid"]

    exp_col, remove_col = st.columns([10, 1])

    with exp_col:
        with st.expander(f"Calculation #{i + 1}", expanded=True):
            name = clean_name(
                st.text_input(
                    "Calculation name",
                    value=item.get("name", f"calculation_{i+1}"),
                    key=f"name_{uid}",
                    help="A separate output subfolder is created using this name.",
                ),
                f"calculation_{i+1}",
            )

            st.caption(f"Results: `{output_folder}/{name}`")

            mode = st.selectbox(
                "Graph calculation mode",
                [
                    "Full graph",
                    "Connected component from root",
                    "Path between two nodes",
                ],
                key=f"mode_{uid}",
            )

            root_node = ""
            start_node = ""
            goal_node = ""

            if mode == "Connected component from root":
                root_node = st.text_input(
                    "Root node",
                    placeholder="ASP-213",
                    key=f"root_{uid}",
                    help="Current OligoComp CLI expects residue name and residue ID, e.g. ASP-213.",
                ).strip()
                if not root_node:
                    st.warning("A root node is required for this calculation.")

            elif mode == "Path between two nodes":
                p1, p2 = st.columns(2)
                start_node = p1.text_input(
                    "Start node",
                    placeholder="GLU-236",
                    key=f"path_start_{uid}",
                ).strip()
                goal_node = p2.text_input(
                    "Goal node",
                    placeholder="ASN-51",
                    key=f"path_goal_{uid}",
                ).strip()

                if start_node and goal_node and start_node == goal_node:
                    st.error("Start and goal nodes must be different.")

            st.subheader("Trajectory and graph parameters")

            c1, c2, c3 = st.columns(3)
            start = c1.number_input(
                "Start frame",
                value=None,
                step=1,
                key=f"start_{uid}",
                help="Blank = first frame. Negative values are accepted by the underlying trajectory reader.",
            )
            stop = c2.number_input(
                "Stop frame",
                value=None,
                step=1,
                key=f"stop_{uid}",
                help="Blank = last frame.",
            )
            step = c3.number_input(
                "Step",
                value=1,
                min_value=1,
                step=1,
                key=f"step_{uid}",
            )

            c1, c2, c3, c4 = st.columns(4)
            distance = c1.number_input(
                "H-bond distance cutoff (Å)",
                value=3.5,
                min_value=0.1,
                max_value=15.0,
                step=0.1,
                key=f"distance_{uid}",
            )
            cut_angle = c2.number_input(
                "H-bond angle cutoff (°)",
                value=60.0,
                min_value=0.0,
                max_value=180.0,
                step=1.0,
                key=f"angle_{uid}",
            )
            occupancy = c3.number_input(
                "Minimum occupancy",
                value=0.1,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key=f"occupancy_{uid}",
            )
            max_water = c4.number_input(
                "Maximum waters",
                value=3,
                min_value=0,
                step=1,
                key=f"maxwater_{uid}",
            )

            conservation_threshold = st.number_input(
                "Conservation threshold",
                value=0.9,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key=f"conservation_{uid}",
                help="Fraction of monomers/segments in which a node or edge must occur to be considered conserved.",
            )

            selection = st.text_input(
                "MDAnalysis selection",
                value="protein",
                key=f"selection_{uid}",
                help="Example: `protein or resname LYR or resname HSE`.",
            )

            segment_input = st.text_input(
                "Segment names (optional)",
                value="",
                key=f"segments_{uid}",
                help="Comma-separated topology segment IDs. Blank = use all protein segments.",
            )
            segment_names = [x.strip() for x in segment_input.split(",") if x.strip()]

            c1, c2, c3 = st.columns(3)
            residuewise = c1.radio(
                "Graph resolution",
                ["Residuewise", "Atomwise"],
                horizontal=True,
                key=f"resolution_{uid}",
            ) == "Residuewise"

            wrap_dcd = c2.selectbox(
                "PBC wrapping",
                [True, False],
                key=f"wrap_{uid}",
                help="Passed to `--wrap_dcd true|false`.",
            )

            res_id_label_shift = c3.number_input(
                "Residue ID label shift",
                value=0,
                step=1,
                key=f"shift_{uid}",
                help=(
                    "The current CLI later casts this argument to int, so this generator "
                    "uses one integer offset for the calculation."
                ),
            )

            include_backbone = st.checkbox(
                "Include backbone ↔ side-chain interactions",
                value=False,
                key=f"backbone_{uid}",
            )
            no_label_plots = st.checkbox(
                "Generate additional plots without labels",
                value=True,
                key=f"nolabel_{uid}",
            )
            dont_save_graph_objects = st.checkbox(
                "Don't save large graph/metadata objects",
                value=True,
                key=f"dontsave_{uid}",
            )
            collect_angles = st.checkbox(
                "Collect H-bond angles",
                value=False,
                key=f"angles_{uid}",
            )
            run_in_background = st.checkbox(
                "Run this calculation in the background",
                value=False,
                key=f"background_{uid}",
                help=(
                    "Background calculations run in parallel. Leave this off if calculations "
                    "are large or you want to limit memory/CPU usage."
                ),
            )

            calculations.append(
                {
                    "uid": uid,
                    "name": name,
                    "mode": mode,
                    "root_node": root_node,
                    "start_node": start_node,
                    "goal_node": goal_node,
                    "start": start,
                    "stop": stop,
                    "step": int(step),
                    "distance": float(distance),
                    "cut_angle": float(cut_angle),
                    "occupancy": float(occupancy),
                    "max_water": int(max_water),
                    "conservation_threshold": float(conservation_threshold),
                    "selection": selection,
                    "segment_names": segment_names,
                    "residuewise": residuewise,
                    "wrap_dcd": wrap_dcd,
                    "res_id_label_shift": int(res_id_label_shift),
                    "include_backbone": include_backbone,
                    "no_label_plots": no_label_plots,
                    "dont_save_graph_objects": dont_save_graph_objects,
                    "collect_angles": collect_angles,
                    "run_in_background": run_in_background,
                }
            )

    with remove_col:
        if len(st.session_state.oligocomp_calculations) > 1:
            if st.button("❌", key=f"remove_{uid}", help="Remove this calculation"):
                st.session_state.oligocomp_calculations = [
                    x for x in st.session_state.oligocomp_calculations
                    if x["uid"] != uid
                ]
                st.rerun()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

errors = []
for i, calc in enumerate(calculations, start=1):
    if calc["mode"] == "Connected component from root" and not calc["root_node"]:
        errors.append(f"Calculation {i}: root node is missing.")
    if calc["mode"] == "Path between two nodes":
        if not calc["start_node"] or not calc["goal_node"]:
            errors.append(f"Calculation {i}: both path nodes are required.")
        elif calc["start_node"] == calc["goal_node"]:
            errors.append(f"Calculation {i}: path start and goal nodes are identical.")

if len({c["name"] for c in calculations}) != len(calculations):
    errors.append("Calculation names must be unique because each name is used as an output folder.")

st.divider()
st.subheader("📄 Live Bash Script Preview")

env_params = {
    "venv_path": venv_path,
    "oligocomp_dir": oligocomp_dir,
    "output_folder": output_folder,
    "run_name": run_name,
    "psf": psf,
    "dcd": dcd,
}

global_params = {
    "donors": donors,
    "acceptors": acceptors,
    "plot_parameters": repr(plot_parameters),
}

run_script = generate_bash(env_params, global_params, calculations)

if errors:
    for error in errors:
        st.error(error)
    st.warning("The preview is shown, but fix the validation errors before using the run file.")

st.code(run_script, language="bash")

st.warning(
    f"After downloading, make the file executable with: `chmod +x run_{run_name}`"
)

st.download_button(
    label="Download OligoComp Run File",
    data=run_script,
    file_name=f"run_{run_name}",
    mime="text/x-shellscript",
    disabled=bool(errors),
)
