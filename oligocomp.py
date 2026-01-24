import helperfunctions as _hf
import numpy as np
import pandas as pd
import MDAnalysis as _mda
import mdhbond as mdh
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse
import glob
import ast
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis.*")


import pdb
class OligoComp:
    def __init__(
        self,
        target_folder,
        psf_file,
        dcd_files,
        segment_names=None,
        plot_parameters={},
        dont_save_graph_objects=False,
    ):

        self.plot_parameters = _hf.get_plot_parameters(plot_parameters)
        self.target_folder = target_folder
        self.workfolder = _hf.create_directory(Path(target_folder, "workfolder"))

        if not dont_save_graph_objects:
            self.graph_object_folder = _hf.create_directory(
                Path(self.workfolder, "graph_objects")
            )

        self.helper_files_folder = _hf.create_directory(
            Path(self.workfolder, ".helper_files")
        )

        self.logger = _hf.create_logger(self.helper_files_folder)

        self.psf_file = psf_file
        self.dcd_files = dcd_files
        if segment_names:
            self.segments = segment_names
        else:
            u = _mda.Universe(psf_file, dcd_files)
            self.segments = u.select_atoms("protein").segments.segids

        self.graph_coord_objects = {}
        for segment in self.segments:
            self.graph_coord_objects[segment] = {
                "psf": self.psf_file,
                "dcd": self.dcd_files,
            }

        self.inter_monomer = (segment_names == ['inter_monomer'])
        self.graph_type = "water_wire"

    def _add_node_positions_from_structure(self, selected_atoms, graph, residuewise):
        node_positions = {}
        for i, residue in enumerate(selected_atoms):
            chain, res_name, res_id = (
                residue.segid,
                residue.resname,
                residue.resid,
            )
            if residuewise:
                res = f"{chain}-{res_name}-{res_id}"
            else:
                atom_name = residue.name
                res = f"{chain}-{res_name}-{res_id}-{atom_name}"

            if res in graph.nodes:
                node_positions.update({res: residue.position})
        return node_positions

    def calculate_graphs(
        self,
        selection="protein",
        max_water=3,
        # exclude_backbone_backbone=True,
        include_backbone_sidechain=False,
        distance=3.5,
        cut_angle=60.0,
        check_angle=True,
        additional_donors=[],
        additional_acceptors=[],
        step=1,
        start=None,
        stop=None,
        residuewise=True,
        wrap_dcd=False,
        connected_component_root=None,
        path=(),
        occupancy=None,
        dont_save_graph_objects=False,
        inter_monomer=False,
        collect_angles=False
    ):
        self.distance = distance
        self.connected_component_root = connected_component_root
        self.path = path
        self.logger.info(f"H-bond criteria cut off distance: {self.distance} A")

        self.include_backbone_sidechain = include_backbone_sidechain

        if include_backbone_sidechain:
            self.logger.info("Including sidechain-backbone interactions")
            additional_donors.append("N")
            additional_acceptors.append("O")

        self.selection = selection
        self.logger.info(f"Atom selection string: {self.selection}")

        self.max_water = max_water
        self.logger.info(
            f"Maximum number of water in water bridges is set to : {self.max_water}"
        )

        if additional_donors or additional_acceptors:
            self.logger.info(
                f"""List of additional donors: {additional_donors}
                List of additional acceptors: {additional_acceptors}
                """
            )

        if check_angle:
            self.logger.info(f"H-bond criteria cut off angle: {cut_angle} degree")

        self.residuewise = residuewise
        self.logger.info(
            f"Calculating H-bonds {'residuewise' if self.residuewise else 'atomwise'}"
        )

        for segment in self.segments:
            self.logger.info(
                f"""Loading {len(self.dcd_files)} trajectory files for segment {segment}.
                From frame {1 if start is None else start} until frame {'last' if stop is None else stop} with a step size of {step}."""
            )
            self.logger.info("This step takes some time...")

            selection = f'({self.selection})' if inter_monomer else f'(segid {segment}) and ({self.selection})'

            psf_file = self.graph_coord_objects[segment]['psf']
            dcd_files = self.graph_coord_objects[segment]['dcd']

            wba = mdh.WireAnalysis(
                selection,
                psf_file,
                dcd_files,
                residuewise=self.residuewise,
                check_angle=check_angle,
                add_donors_without_hydrogen=not check_angle,
                additional_donors=additional_donors,
                additional_acceptors=additional_acceptors,
                distance=distance,
                cut_angle=cut_angle,
                wrap_dcd=wrap_dcd,
                step=step,
                start=start,
                stop=stop,
            )

            angles_per_frame = wba.set_water_wires(water_in_convex_hull=max_water, max_water=max_water)

            wba.compute_average_water_per_wire()
            if connected_component_root:
                self.logger.info(f"Performing connected component search from root: {connected_component_root}")
                seg_id, res_name, res_id = _hf.get_node_name_pats(connected_component_root)
                root = f'{segment}-{res_name}-{res_id}'
                if occupancy:
                    wba.filter_occupancy(occupancy)
                wba.filter_connected_component(root)

            elif path:
                self.logger.info(f"Performing path search between start {path[0]} and goal {path[1]}.")
                seg_id, res_name, res_id = _hf.get_node_name_pats(path[0])
                start_node = f'{segment}-{res_name}-{res_id}'

                seg_id, res_name, res_id = _hf.get_node_name_pats(path[1])
                goal_node = f'{segment}-{res_name}-{res_id}'
                if occupancy:
                    wba.filter_occupancy(occupancy)
                if segment == 'inter_monomer':
                    wba.filter_inter_monomer_paths(start_node, goal_node)
                else:
                    wba.filter_all_paths(start_node, goal_node)

            if len(wba.filtered_graph.nodes) == 0:
                self.logger.warning(f"No H-bond network was found. The graph is empty with the provided criteria.")
                continue
            self.graph_coord_objects[segment].update({"wba": wba})

            self.graph = wba.filtered_graph
            self.graph_coord_objects[segment].update({"graph": self.graph})

            u = _mda.Universe(psf_file, dcd_files)

            if inter_monomer:
                selection = f'({self.selection})'
                self.multi_segments = [
                    seg.segid for seg in u.select_atoms("protein").segments
                ]
            else:
                selection = f'(segid {segment}) and ({self.selection})'
                self.multi_segments = []

            selected_atoms = u.select_atoms(selection)
            self.graph_coord_objects[segment].update({"selected_atoms": selected_atoms})

            if self.connected_component_root:
                plot_folder = _hf.create_directory(
                    Path(self.workfolder, f"{self.max_water}_water_wires_connected_components", self.connected_component_root, segment)
                )
            elif self.path:
                path_name = f'{path[0]}-{path[1]}'
                plot_folder = _hf.create_directory(
                    Path(self.workfolder, f"{self.max_water}_water_wires_path", path_name, segment)
                )
            else:
                plot_folder = _hf.create_directory(
                    Path(self.workfolder, f"{self.max_water}_water_wires", segment)
                )

            if collect_angles:
                no_self_pair = [col for col in angles_per_frame.columns if not col.split('_')[0].split("-")[:3] == col.split('_')[1].split("-")[:3]]
                angles_per_frame[no_self_pair].to_csv(Path(plot_folder, f"{segment}_angles.csv"), index=False),

            df = pd.DataFrame.from_dict(
                _hf.edge_info(wba, self.graph.edges), orient="index"
            ).reset_index()
            df.columns = ["edge", "water", "occupancy"]
            df['edge'] = df['edge'].str.replace(':', '_')
            root = (f"_{self.connected_component_root}" if self.connected_component_root else "")
            path_name = f'_path_{self.path[0]}-{self.path[1]}' if self.path else ""

            waters = f"_max_{self.max_water}_water_bridges"
            df.to_csv(
                Path(plot_folder, f"{segment}{root}{path_name}{waters}_water_occupancy_all_edge_info.txt"),
                sep="\t",
                index=False,
            )
            if occupancy:
                df[df['occupancy'] >= occupancy].to_csv(
                    Path(plot_folder, f"{segment}{root}{path_name}{waters}_water_{occupancy}_occupancy_edge_info.txt"),
                    sep="\t",
                    index=False,
                )

            if not dont_save_graph_objects:
                if self.connected_component_root:
                    root = f"_{self.connected_component_root}_"
                    path_name = ""
                    self.water_graphs_folder = _hf.create_directory(
                        Path(
                            self.graph_object_folder,
                            f"{self.max_water}_water_wires_connected_components",
                        )
                    )
                elif self.path:
                    path_name = f'{path[0]}-{path[1]}'
                    root = ""
                    self.water_graphs_folder = _hf.create_directory(
                        Path(
                            self.graph_object_folder,
                            f"{self.max_water}_water_wires_path",
                        )
                    )

                else:
                    root = ""
                    path_name = ""
                    self.water_graphs_folder = _hf.create_directory(
                        Path(self.graph_object_folder, f"{self.max_water}_water_wires")
                    )

                wba.dump_to_file(
                    Path(
                        self.water_graphs_folder,
                        f"{segment}_{root}{path_name}{self.max_water}_water_wires_graph.pickle",
                    )
                )

                _hf.pickle_write_file(
                    Path(
                        self.helper_files_folder,
                        f"{segment}_{root}{path_name}{self.max_water}_water_nx_graphs.pickle",
                    ),
                    self.graph,
                )

                _hf.json_write_file(
                    Path(
                        self.helper_files_folder,
                        f"{segment}_{root}{path_name}{self.max_water}_water_graph_edge_info.json",
                    ),
                    _hf.edge_info(wba, self.graph.edges),
                )

                graph_coord_object_loc = Path(
                    self.helper_files_folder,
                    f"{segment}_{root}{path_name}{self.max_water}_water_wires_coord_objects.pickle",
                )
                _hf.pickle_write_file(
                    graph_coord_object_loc,
                    self.graph_coord_objects[segment],
                )
                self.logger.info(f"Graph object is saved as: {graph_coord_object_loc}")

            self.node_positions = self._add_node_positions_from_structure(
                selected_atoms, self.graph, self.residuewise
            )
            self.pca_positions = _hf.calculate_pca_positions(self.node_positions)

    def _get_node_positions(self, segment, pca=True):
        node_pos = {}
        for node in segment["graph"].nodes:
            n = _hf.get_node_name(node, with_segid=self.inter_monomer)
            if (
                n not in self.node_positions.keys()
                or n.split("-")[1] in _hf.water_types
            ):
                seg_id, res_name, res_id = _hf.get_node_name_pats(n, with_segid=self.inter_monomer)
                coords = (
                    segment["selected_atoms"]
                    .select_atoms("resid " + res_id)
                    .positions[0]
                )
                if coords is not None:
                    node_pos.update({n: list(coords)})
            else:
                node_pos.update({n: self.node_positions[n]})
        if pca:
            return _hf.calculate_pca_positions(node_pos)
        else:
            return node_pos

    def filter_inter_monomer(self, graph):
        inter_monomer_edges = [(e1, e2) for e1, e2 in graph.edges() if e1.split('-')[0] != e2.split('-')[0]]
        return graph.edge_subgraph(inter_monomer_edges).copy()

    def find_conserved_inter_monomer_edges(self, graph):
        graph = self.filter_inter_monomer(graph)
        inter_monomer_edges = []
        monomers = []
        for edge in graph.edges:
            monomers.append(_hf.get_node_name(edge[0], with_segid=self.inter_monomer).split('-')[0])
            monomers.append(_hf.get_node_name(edge[1], with_segid=self.inter_monomer).split('-')[0])

            e0 = ('-').join(_hf.get_node_name(edge[0], with_segid=self.inter_monomer).split('-')[1:])
            e1 = ('-').join(_hf.get_node_name(edge[1], with_segid=self.inter_monomer).split('-')[1:])

            inter_monomer_edges.append(sorted((e0, e1)))

        u_monomers = np.unique(monomers)
        u_edges, c_edges = np.unique(inter_monomer_edges, axis=0, return_counts=True)
        conserved_inter_monomer_edges = u_edges[np.where(c_edges == len(u_monomers))[0]]
        return conserved_inter_monomer_edges

    def get_conserved_graph(self, conservation_threshold=1, occupancy=None, eps=1.5):
        self.logger.info(
            "Conservation threshold across structures is set to: "
            + str(conservation_threshold * 100)
            + "%"
        )
        if occupancy:
            self.logger.info(
                "H-bond occupancy is set to: " + str(occupancy * 100) + "%"
            )
        nodes = []
        edges = []
        self.avg_water_per_conserved_edges = None
        avg_water_per_edge = {}

        for objects in self.graph_coord_objects.values():
            if "graph" in objects.keys():
                avg_waters = objects["wba"].compute_average_water_per_wire()
                if occupancy:
                    # wba = copy.deepcopy(objects["wba"])
                    wba = objects["wba"]
                    wba.filter_occupancy(occupancy)
                    graph = wba.filtered_graph
                else:
                    graph = objects["graph"]
                for node in graph.nodes:
                    node = _hf.get_node_name(node)
                    nodes.append(node)

                for edge in graph.edges:
                    e0 = _hf.get_node_name(edge[0])
                    e1 = _hf.get_node_name(edge[1])
                    if ([e1, e0]) in edges:
                        edges.append([e1, e0])
                    else:
                        edges.append([e0, e1])

                    key = e0 + ":" + e1
                    key2 = e1 + ":" + e0

                    if key in avg_waters:
                        if key in avg_water_per_edge:
                            avg_water_per_edge[key].append(avg_waters[key])
                        elif key2 in avg_water_per_edge:
                            avg_water_per_edge[key2].append(avg_waters[key])
                        else:
                            avg_water_per_edge.update({key: [avg_waters[key]]})
                    elif key2 in avg_waters:
                        if key in avg_water_per_edge:
                            avg_water_per_edge[key].append(avg_waters[key2])
                        elif key2 in avg_water_per_edge:
                            avg_water_per_edge[key2].append(avg_waters[key2])
                        else:
                            avg_water_per_edge.update({key2: [avg_waters[key2]]})

        th = np.round(len(self.graph_coord_objects) * conservation_threshold)
        u_nodes, c_nodes = np.unique(nodes, return_counts=True)
        conserved_nodes = u_nodes[np.where(c_nodes >= th)[0]]
        u_edges, c_edges = np.unique(edges, return_counts=True, axis=0)
        self.conserved_edges = u_edges[np.where(c_edges >= th)[0]]
        self.conserved_nodes = np.array(
            [
                n
                for n in conserved_nodes
                if n.split("-")[1] not in _hf.water_types
                or any(n in pair for pair in self.conserved_edges)
            ],
            dtype=str,
        )
        if len(avg_water_per_edge) > 0:
            self.avg_water_per_conserved_edges = {
                key: np.mean(value)
                for key, value in avg_water_per_edge.items()
                if len(value) >= th
            }

    def plot_graphs(
        self,
        label_nodes=True,
        label_edges=True,
        xlabel="PCA projected membrane plane",
        ylabel="Membrane normal (Å)",
        occupancy=None,
        inter_monomer=False
    ):
        for segment in self.segments:
            if "wba" not in self.graph_coord_objects[segment]:
                return

            wba = self.graph_coord_objects[segment]["wba"]
            if occupancy:
                wba.filter_occupancy(occupancy)
                graph = wba.filtered_graph
            else:
                graph = self.graph_coord_objects[segment]["graph"]

            self.logger.debug(f"Creating water wire graph for {segment}")
            fig, ax = _hf.create_plot(
                title=f"""Water wire graph of structure {segment}
                Selection: {self.selection}""",
                xlabel=xlabel,
                ylabel=ylabel,
                plot_parameters=self.plot_parameters,
            )
            node_pca_pos = self._get_node_positions(self.graph_coord_objects[segment])
            node_pca_pos = _hf.check_projection_sign(node_pca_pos, self.pca_positions)

            inter_monomer_edges = []
            conserved_inter_monomer_edges = []
            unique_inter_monomer_edges = []
            conserved_inter_monomer_edges = self.find_conserved_inter_monomer_edges(graph)

            if inter_monomer:
                graph = self.filter_inter_monomer(graph)
                self.plot_parameters["show_chain_label"] = True

            for e in graph.edges:
                e0 = _hf.get_node_name(e[0], with_segid=self.inter_monomer)
                e1 = _hf.get_node_name(e[1], with_segid=self.inter_monomer)
                if e0 in node_pca_pos.keys() and e1 in node_pca_pos.keys():
                    edge_line = [node_pca_pos[e0], node_pca_pos[e1]]
                    x = [edge_line[0][0], edge_line[1][0]]
                    y = [edge_line[0][1], edge_line[1][1]]

                    sorted_e_pair_no_segid = sorted((('-').join(_hf.get_node_name(e[0], with_segid=self.inter_monomer).split('-')[1:]), ('-').join(_hf.get_node_name(e[1], with_segid=self.inter_monomer).split('-')[1:])))

                    if self.inter_monomer and len(conserved_inter_monomer_edges) and np.any(np.all(sorted_e_pair_no_segid == conserved_inter_monomer_edges, axis=1)):
                        color = 'blue'
                        inter_monomer_edges.append(e)

                    elif self.inter_monomer and e0.split('-')[0] != e1.split('-')[0]:
                        color = 'darkorange'  # 'darkorange'
                        inter_monomer_edges.append(e)
                        unique_inter_monomer_edges.append(e)
                    else:
                        color = 'lightgray'  # self.plot_parameters["graph_color"],

                    ax.plot(
                        x,
                        y,
                        color=color,
                        marker="o",
                        linewidth=self.plot_parameters["edge_width"],
                        markersize=self.plot_parameters["node_size"] * 0.01,
                        markerfacecolor=self.plot_parameters["graph_color"],
                        markeredgecolor=self.plot_parameters["graph_color"],
                    )

                    if label_nodes:
                        waters, occ_per_wire, _ = _hf.get_edge_params(wba, graph.edges)
                        ax.annotate(
                            np.round(waters[list(graph.edges).index(e)], 1),
                            (x[0] + (x[1] - x[0]) / 2, y[0] + (y[1] - y[0]) / 2),
                            color="indianred",
                            fontsize=self.plot_parameters["edge_label_size"],
                            weight="bold",
                        )
                        if occupancy:
                            ax.annotate(
                                int(occ_per_wire[list(graph.edges).index(e)] * 100),
                                (
                                    x[0] + (x[1] - x[0]) / 2,
                                    y[0] + (y[1] - 1.0 - y[0]) / 2,
                                ),
                                color="green",
                                fontsize=self.plot_parameters["edge_label_size"],
                            )

            markers = ["o", ",", "v", "p", "D", "*", "h", "H", "X"]
            for n in graph.nodes:
                if self.multi_segments:
                    marker_shape = markers[self.multi_segments.index(n.split("-")[0])]
                else:
                    marker_shape = "o"

                n = _hf.get_node_name(n, with_segid=self.inter_monomer)
                if n in node_pca_pos.keys():
                    values = node_pca_pos[n]
                    segid, res_name, res_id = _hf.get_node_name_pats(n, with_segid=self.inter_monomer)
                    if res_name in _hf.water_types:
                        ax.scatter(
                            values[0],
                            values[1],
                            marker=marker_shape,
                            color=self.plot_parameters["water_node_color"],
                            s=self.plot_parameters["node_size"] * 0.7,
                            zorder=5,
                        )
                    elif res_name in _hf.amino_d.keys():
                        ax.scatter(
                            values[0],
                            values[1],
                            marker=marker_shape,
                            color=self.plot_parameters["graph_color"],
                            s=self.plot_parameters["node_size"],
                            zorder=5,
                            edgecolors=self.plot_parameters["graph_color"],
                        )
                    else:
                        ax.scatter(
                            values[0],
                            values[1],
                            marker=marker_shape,
                            color=self.plot_parameters["non_prot_color"],
                            s=self.plot_parameters["node_size"],
                            zorder=5,
                            edgecolors=self.plot_parameters["graph_color"],
                        )

            if label_nodes:
                for n in graph.nodes:
                    n = _hf.get_node_name(n, with_segid=self.inter_monomer)
                    if n in node_pca_pos.keys():
                        values = node_pca_pos[n]
                        seg_id, res_name, res_id = _hf.get_node_name_pats(n,  with_segid=self.inter_monomer)
                        if res_name in _hf.water_types:
                            pass  # temporary turn off water labels
                            # ax.annotate(
                            #     f"W{res_id}",
                            #     (values[0] + 0.2, values[1] - 0.25),
                            #     fontsize=self.plot_parameters["node_label_size"],
                            # )
                        elif res_name in _hf.amino_d.keys():
                            res_label = (
                                f"{seg_id}-{_hf.amino_d[res_name]}{res_id}"
                                if self.plot_parameters["show_chain_label"]
                                else f"{_hf.amino_d[res_name]}{res_id}"
                            )

                            ax.annotate(
                                res_label,
                                (values[0] + 0.26, values[1] - 0.26),
                                fontsize=self.plot_parameters["node_label_size"],
                            )
                        else:
                            res_label = (
                                f"{seg_id}-{res_name}{res_id}"
                                if self.plot_parameters["show_chain_label"]
                                else f"{res_name}{res_id}"
                            )
                            ax.annotate(
                                res_label,
                                (values[0] + 0.26, values[1] - 0.25),
                                fontsize=self.plot_parameters["node_label_size"],
                                color=self.plot_parameters["non_prot_color"],
                            )

            plt.tight_layout()
            is_label = "_labeled" if label_nodes else ""
            is_backbone = (
                "_backbone"
                if hasattr(self, "include_backbone_sidechain")
                and self.include_backbone_sidechain
                else ""
            )

            inter_monomer = '_inter_monomer' if inter_monomer else ''

            if self.connected_component_root:
                plot_folder = _hf.create_directory(
                    Path(self.workfolder, f"{self.max_water}_water_wires_connected_components", self.connected_component_root, segment)
                )
            elif self.path:
                plot_folder = _hf.create_directory(
                    Path(self.workfolder, f"{self.max_water}_water_wires_path", f'{self.path[0]}-{self.path[1]}', segment)
                )
            else:
                plot_folder = _hf.create_directory(
                    Path(self.workfolder, f"{self.max_water}_water_wires", segment)
                )

            waters = f"_max_{self.max_water}_water_bridges"
            occ = f"_min_occupancy_{occupancy}" if occupancy else ""
            root = (
                f"_cc_{self.connected_component_root}" if self.connected_component_root else ""
            )
            path_name = f'_path_{self.path[0]}-{self.path[1]}' if self.path else ""
            for form in self.plot_parameters["formats"]:
                plt.savefig(
                    Path(
                        plot_folder,
                        f"{segment}{root}{path_name}{waters}{occ}_graph{is_backbone}{inter_monomer}{is_label}.{form}",
                    ),
                    format=form,
                    dpi=self.plot_parameters["plot_resolution"],
                )
            if self.inter_monomer and len(inter_monomer_edges):
                _hf.write_text_file(
                    Path(
                        plot_folder,
                        f"{segment}{root}{path_name}{waters}{occ}_inter_monomer_edges.txt",
                    ),
                    [
                        f"Water wire graph of {segment}",
                        f"\nSelection string: {self.selection[0:-15]}",
                        f"\nNumber of maximum water molecules allowed in the bridge: {self.max_water}",
                        (
                            f"\nMinimum H-bond occupancy: {occupancy}"
                            if occupancy
                            else ""
                        ),
                        (f"\nConnected component from root node {root}" if root else ""),
                        (f"\nPath search between {self.path[0]} and {self.path[1]}" if self.path else ""),
                        "\n",
                        f"\nNumber of inter monomer edges: {len(inter_monomer_edges)}"
                        f"\nList of edges: {sorted(inter_monomer_edges)}",
                        "\n",
                        f"\nNumber of conserved inter monomer edges: {len(conserved_inter_monomer_edges)}",
                        f"\nList of conserved inter monomer edges: {conserved_inter_monomer_edges}",
                        "\n",
                        f"\nList of NOT conserved inter monomer edges: {unique_inter_monomer_edges}",
                    ],
                )
            if is_label:
                _hf.write_text_file(
                    Path(
                        plot_folder,
                        f"{segment}{root}{path_name}{waters}{occ}_water_wire_graph_info.txt",
                    ),
                    [
                        "Water wire graph of " + segment,
                        "\nSelection string: " + str(self.selection[0:-15]),
                        "\nNumber of maximum water molecules allowed in the bridge: "
                        + str(self.max_water),
                        (
                            "\nMinimum H-bond occupancy: " + str(occupancy)
                            if occupancy
                            else ""
                        ),
                        (f"\nConnected component from root node {root}" if root else ""),
                        (f"\nPath search between {self.path[0]} and {self.path[1]}" if self.path else ""),
                        "\n",
                        "\nNumber of nodes in "
                        + segment
                        + ": "
                        + str(len(graph.nodes)),
                        "\nNumber of edges in "
                        + segment
                        + ": "
                        + str(len(graph.edges)),
                        "\n",
                        "\nList of nodes: " + str(graph.nodes),
                        "\n",
                        "\nList of edges: " + str(graph.edges),
                    ],
                )
            plt.close()

    def plot_conserved_graph(
        self,
        label_nodes=True,
        label_edges=True,
        xlabel="PCA projected membrane plane",
        ylabel="Membrane normal (Å)",
        conservation_threshold=0.9,
        occupancy=None
    ):
        self.logger.info(
            "Plotting conserved "
            + self.graph_type
            + " graph"
            + str(" with labels" if label_nodes else "")
        )

        self.get_conserved_graph(conservation_threshold, occupancy)

        index_of_has_graph = None
        for i, seg in enumerate(self.segments):
            if 'graph' in self.graph_coord_objects[seg]:
                index_of_has_graph = i
                break

        if index_of_has_graph is None:
            return


        if len(self.conserved_nodes) == 0:
            self.logger.warning(f"No conserved H-bond network was found. The conserved graph is empty with the provided criteria.")
            return
        self.pca_positions = self._get_node_positions(self.graph_coord_objects[self.segments[0]])

        plot_name = "H-bond" if self.graph_type == "hbond" else "water wire"
        fig, ax = _hf.create_plot(
            title=f"Conserved {plot_name} graph\nSelection: {self.selection}",
            xlabel=xlabel,
            ylabel=ylabel,
            plot_parameters=self.plot_parameters,
        )
        for e in self.conserved_edges:
            if e[0] in self.pca_positions.keys() and e[1] in self.pca_positions.keys():
                edge_line = [self.pca_positions[e[0]], self.pca_positions[e[1]]]
                x = [edge_line[0][0], edge_line[1][0]]
                y = [edge_line[0][1], edge_line[1][1]]
                ax.plot(
                    x,
                    y,
                    color=self.plot_parameters["graph_color"],
                    marker="o",
                    linewidth=self.plot_parameters["edge_width"],
                    markersize=self.plot_parameters["node_size"] * 0.01,
                    markerfacecolor=self.plot_parameters["graph_color"],
                    markeredgecolor=self.plot_parameters["graph_color"],
                )
            if label_nodes and self.avg_water_per_conserved_edges:
                key1 = e[0] + ":" + e[1]
                key2 = e[1] + ":" + e[0]
                water = [
                    value
                    for key, value in self.avg_water_per_conserved_edges.items()
                    if key == key1 or key == key2
                ][0]
                ax.annotate(
                    np.round(water, 1),
                    (x[0] + (x[1] - x[0]) / 2, y[0] + (y[1] - y[0]) / 2),
                    color="indianred",
                    fontsize=self.plot_parameters["edge_label_size"],
                    weight="bold",
                )

        for n in self.conserved_nodes:
            if n in self.pca_positions.keys():
                color = (
                    self.plot_parameters["water_node_color"]
                    if n.split("-")[1] in _hf.water_types
                    else self.plot_parameters["graph_color"]
                )
                ax.scatter(
                    self.pca_positions[n][0],
                    self.pca_positions[n][1],
                    color=color,
                    s=self.plot_parameters["node_size"],
                    zorder=5,
                )

        if label_nodes:
            for node in self.conserved_nodes:
                seg_id, res_name, res_id = _hf.get_node_name_pats(node, with_segid=self.inter_monomer)
                if node in self.pca_positions.keys():
                    if (
                        res_name not in _hf.water_types
                        and res_name in _hf.amino_d.keys()
                    ):
                        l = (
                            f"{segment}-{_hf.amino_d[res_name]}{res_id}"
                            if self.plot_parameters["show_chain_label"]
                            else f"{_hf.amino_d[res_name]}{res_id}"
                        )
                        ax.annotate(
                            l,
                            (
                                self.pca_positions[node][0] + 0.2,
                                self.pca_positions[node][1] - 0.25,
                            ),
                            fontsize=self.plot_parameters["node_label_size"],
                            zorder=6,
                        )
                    elif (
                        res_name not in _hf.water_types
                        and res_name not in _hf.amino_d.keys()
                    ):
                        l = (
                            f"{segment}-{res_name}{res_id}"
                            if self.plot_parameters["show_chain_label"]
                            else f"{res_name}{res_id}"
                        )
                        ax.annotate(
                            l,
                            (
                                self.pca_positions[node][0] + 0.2,
                                self.pca_positions[node][1] - 0.25,
                            ),
                            fontsize=self.plot_parameters["node_label_size"],
                            zorder=6,
                            color=self.plot_parameters["non_prot_color"],
                        )

        plt.tight_layout()
        is_label = "_labeled" if label_nodes else ""
        is_backbone = (
            "_backbone"
            if hasattr(self, "include_backbone_sidechain")
            and self.include_backbone_sidechain
            else ""
        )
        if self.connected_component_root:
            plot_folder = _hf.create_directory(
                Path(self.workfolder, f"{self.max_water}_water_wires_connected_components", self.connected_component_root)
            )
        elif self.path:
            plot_folder = _hf.create_directory(
                Path(self.workfolder, f"{self.max_water}_water_wires_path", f'{self.path[0]}-{self.path[1]}')
            )
        else:
            plot_folder = _hf.create_directory(
                Path(self.workfolder, f"{self.max_water}_water_wires")
            )

        waters = f"_max_{self.max_water}_water_bridges"
        occ = f"_min_occupancy_{occupancy}" if occupancy else ""
        root = (
            f"_{self.connected_component_root}" if self.connected_component_root else ""
        )
        path_name = f'_path_{self.path[0]}-{self.path[1]}' if self.path else ""

        for form in self.plot_parameters["formats"]:
            plt.savefig(
                Path(plot_folder, f'conserved{root}{waters}{path_name}{occ}_graph{is_backbone}{is_label}.{form}'),
                format=form,
                dpi=self.plot_parameters["plot_resolution"],
            )
        if is_label:
            _hf.write_text_file(
                Path(plot_folder, f'conserved{root}{path_name}{waters}{occ}_graph_inof.txt'),
                [
                    "Conserved water wire graph of "
                    + str(len(self.graph_coord_objects.keys()))
                    + str(" monomers"),
                    "\nSelection string: " + str(self.selection[0:-15]),
                    "\nNumber of maximum water molecules allowed in the bridge: "
                    + str(self.max_water),
                    (
                        "\nMinimum H-bond occupancy: " + str(occupancy)
                        if occupancy
                        else ""
                    ),
                    (f"\nConnected component from root node {root}" if root else ""),
                    (f"\nPath search between {self.path[0]} and {self.path[1]}" if self.path else ""),
                    "\n",
                    "\nNumber of conserved nodes : "
                    + str(len(self.conserved_nodes)),
                    "\nNumber of conserved edges : "
                    + str(len(self.conserved_edges)),
                    "\n",
                    "\nList of conserved nodes: " + str(self.conserved_nodes),
                    "\n",
                    "\nList of conserved edges: " + str(self.conserved_edges),
                ],
            )
        plt.close()

    def plot_differnece_graphs(
        self,
        label_nodes=True,
        label_edges=True,
        xlabel="PCA projected membrane plane",
        ylabel="Membrane normal (Å)",
        conservation_threshold=0.9,
        occupancy=None
    ):
        self.logger.info(
            "Plotting difference "
            + self.graph_type
            + " graphs"
            + str(" with labels" if label_nodes else "")
        )
        for segment, objects in self.graph_coord_objects.items():
            if "graph" in objects.keys():
                if occupancy:
                    wba = objects["wba"]
                    wba.filter_occupancy(occupancy)
                    graph = wba.filtered_graph
                else:
                    graph = objects["graph"]

                self.logger.debug(
                    "Calculating " + self.graph_type + " difference graph for: " + segment
                )
                plot_name = "H-bond" if self.graph_type == "hbond" else "water wire"
                fig, ax = _hf.create_plot(
                    title=f"Difference {plot_name} graph of segment {segment}\nSelection: {self.selection}",
                    xlabel=xlabel,
                    ylabel=ylabel,
                    plot_parameters=self.plot_parameters,
                )
                node_pca_pos = self._get_node_positions(objects)
                node_pca_pos = _hf.check_projection_sign(
                    node_pca_pos, self.pca_positions
                )

                for e in graph.edges:
                    e0 = _hf.get_node_name(e[0])
                    e1 = _hf.get_node_name(e[1])
                    if e0 in node_pca_pos.keys() and e1 in node_pca_pos.keys():
                        edge_line = [node_pca_pos[e0], node_pca_pos[e1]]
                        x = [edge_line[0][0], edge_line[1][0]]
                        y = [edge_line[0][1], edge_line[1][1]]

                        if _hf.is_conserved_edge(self.conserved_edges, e0, e1):
                            ax.plot(
                                x,
                                y,
                                color=self.plot_parameters["graph_color"],
                                marker="o",
                                linewidth=self.plot_parameters["edge_width"],
                                markersize=self.plot_parameters["node_size"] * 0.01,
                                markerfacecolor=self.plot_parameters["graph_color"],
                                markeredgecolor=self.plot_parameters["graph_color"],
                            )
                        else:
                            ax.plot(
                                x,
                                y,
                                color=self.plot_parameters["difference_graph_color"],
                                marker="o",
                                linewidth=self.plot_parameters["edge_width"],
                                markersize=self.plot_parameters["node_size"] * 0.01,
                                markerfacecolor=self.plot_parameters[
                                    "difference_graph_color"
                                ],
                                markeredgecolor=self.plot_parameters[
                                    "difference_graph_color"
                                ],
                            )
                        if label_nodes and self.graph_type == "water_wire":
                            waters, occ_per_wire, _ = _hf.get_edge_params(
                                objects["wba"], graph.edges
                            )
                            ax.annotate(
                                np.round(waters[list(graph.edges).index(e)], 1),
                                (x[0] + (x[1] - x[0]) / 2, y[0] + (y[1] - y[0]) / 2),
                                color="indianred",
                                fontsize=self.plot_parameters["edge_label_size"],
                                weight="bold",
                            )
                            ax.annotate(
                                int(occ_per_wire[list(graph.edges).index(e)] * 100),
                                (
                                    x[0] + (x[1] - x[0]) / 2,
                                    y[0] + (y[1] - 1.0 - y[0]) / 2,
                                ),
                                color="green",
                                fontsize=self.plot_parameters["edge_label_size"],
                            )

                for node in graph.nodes:
                    n = _hf.get_node_name(node)
                    if n in node_pca_pos.keys():
                        if n in self.conserved_nodes:
                            ax.scatter(
                                node_pca_pos[n][0],
                                node_pca_pos[n][1],
                                s=self.plot_parameters["node_size"],
                                color=self.plot_parameters["graph_color"],
                                zorder=5,
                            )
                        else:
                            ax.scatter(
                                node_pca_pos[n][0],
                                node_pca_pos[n][1],
                                s=self.plot_parameters["node_size"],
                                color=self.plot_parameters["difference_graph_color"],
                            )

                for n, values in node_pca_pos.items():
                    if n.split("-")[1] in _hf.water_types:
                        ax.scatter(
                            values[0],
                            values[1],
                            color=self.plot_parameters["water_node_color"],
                            s=self.plot_parameters["node_size"] * 0.7,
                            zorder=5,
                        )

                if label_nodes:
                    for n in graph.nodes:
                        n = _hf.get_node_name(n)
                        if n in node_pca_pos.keys():
                            values = node_pca_pos[n]
                            seg_id, res_name, res_id = _hf.get_node_name_pats(n)
                            if res_name in _hf.water_types:
                                pass  # ax.annotate(f'W{res_id}', (values[0]+0.2, values[1]-0.25), fontsize=self.plot_parameters['node_label_size'])
                            elif res_name in _hf.amino_d.keys():
                                l = (
                                    f"{segment}-{_hf.amino_d[res_name]}{res_id}"
                                    if self.plot_parameters["show_chain_label"]
                                    else f"{_hf.amino_d[res_name]}{res_id}"
                                )
                                ax.annotate(
                                    l,
                                    (values[0] + 0.25, values[1] - 0.25),
                                    fontsize=self.plot_parameters["node_label_size"],
                                )
                            else:
                                l = (
                                    f"{segment}-{res_name}{res_id}"
                                    if self.plot_parameters["show_chain_label"]
                                    else f"{res_name}{res_id}"
                                )
                                ax.annotate(
                                    l,
                                    (values[0] + 0.2, values[1] - 0.25),
                                    fontsize=self.plot_parameters["node_label_size"],
                                    color=self.plot_parameters["non_prot_color"],
                                )

                plt.tight_layout()
                is_label = "_labeled" if label_nodes else ""
                is_backbone = (
                    "_backbone"
                    if hasattr(self, "include_backbone_sidechain")
                    and self.include_backbone_sidechain
                    else ""
                )

                if self.connected_component_root:
                    plot_folder = _hf.create_directory(
                        Path(self.workfolder, f"{self.max_water}_water_wires_connected_components", self.connected_component_root, segment)
                    )
                elif self.path:
                    plot_folder = _hf.create_directory(
                        Path(self.workfolder, f"{self.max_water}_water_wires_path", f'{self.path[0]}-{self.path[1]}', segment)
                    )
                else:
                    plot_folder = _hf.create_directory(
                        Path(self.workfolder, f"{self.max_water}_water_wires", segment)
                    )

                waters = (
                    "_max_" + str(self.max_water) + "_water_bridges"
                    if self.max_water > 0
                    else ""
                )
                occ = (
                    "_min_occupancy_" + str(occupancy)
                    if occupancy
                    else ""
                )
                root = (f"_{self.connected_component_root}" if self.connected_component_root else "")
                path_name = f'_path_{self.path[0]}-{self.path[1]}' if self.path else ""
                for form in self.plot_parameters["formats"]:
                    plt.savefig(
                        Path(plot_folder, f"{segment}{waters}{root}{path_name}{occ}_difference_graph{is_backbone}{is_label}.{form}"),
                        format=form,
                        dpi=self.plot_parameters["plot_resolution"],
                    )
                plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze molecular dynamics trajectories and generate water wire graphs."
    )
    parser.add_argument(
        "psf",
        help="Path to the PSF (Protein Structure File) used for molecular dynamics simulations.",
    )
    parser.add_argument(
        "dcd",
        nargs="+",
        help="Path(s) to the DCD (trajectory) file(s). Supports wildcard patterns (e.g., '*.dcd').",
    )
    parser.add_argument(
        "--segment_names",
        nargs="+",
        help="Names of the segments as given in the topology. These are the different segments or chains between the conserved and difference graphs will be calculated. If non is given, all protein segments will be used.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Directory where output files and plots will be saved. If not provided, defaults to the location of the PSF file.",
    )

    parser.add_argument(
        "--max_water",
        type=int,
        default=3,
        help="Maximum number of water molecules allowed in water wire connections (default: 3).",
    )
    parser.add_argument(
        "--occupancy",
        type=float,
        default=0.1,
        help="Minimum hydrogen bond occupancy required to include an edge in the graph (default: 0.1, which means 10% occupancy).",
    )
    parser.add_argument(
        "--conservation_threshold",
        type=float,
        default=0.9,
        help="TODO",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=3.5,
        help="The distance criterion for the  H-bond search, measured between the heavy atoms. The default value is 3.5Å.",
    )
    parser.add_argument(
        "--cut_angle",
        type=float,
        default=60,
        help="Threshold value for the angle formed by the acceptor heavy atom, the H atom, and the donor heavy atom. The default value is 60°.",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="protein",
        help="Atom selection string for defining the region of interest in the molecular system for graph calculation (default: 'protein').",
    )

    parser.add_argument(
        "--additional_donors",
        type=str,
        default="[]",
        help="""List of additional hydrogen bond donor atoms, formatted as a Python list (e.g., "['N', 'S']").""",
    )
    parser.add_argument(
        "--additional_acceptors",
        type=str,
        default="[]",
        help="""List of additional hydrogen bond acceptor atoms, formatted as a Python list (e.g., "['O', 'F']").""",
    )

    parser.add_argument(
        "--start",
        type=int,
        help="Starting frame index for trajectory analysis. If not provided, starts from the first frame.",
    )
    parser.add_argument(
        "--stop",
        type=int,
        help="Stopping frame index for trajectory analysis. If not provided, processes until the last frame.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size for iterating through the trajectory frames. For example, '--step 10' "
        "processes every 10th frame to reduce computation time. (Default is 1).",
    )
    parser.add_argument(
        "--residuewise",
        default=True,
        action="store_true",
        help="Calculate hydrogen bonds at the residue level instead of the atomic level (default: True).",
    )

    parser.add_argument(
        "--atomewise",
        dest="residuewise",
        action="store_false",
        help="Calculate hydrogen bonds at the atomic level instead of the residue level (overrides --residuewise).",
    )

    parser.add_argument(
        "--wrap_dcd",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Apply periodic boundary condition wrapping to keep molecules inside the simulation box. Use 'true' or 'false' (default: true).",
    )

    parser.add_argument(
        "--plot_parameters",
        default="{}",
        help="""Dictionary of plot parameters formatted as a string : {'graph_color': '#666666', 'formats': ['png', 'eps']}" """,
    )

    parser.add_argument(
        "--include_backbone",
        action="store_true",
        help="Include interactions between backbone and sidechain atoms in the analysis.",
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Root node for connected component search in a form of: ASP-213",
    )

    parser.add_argument( #TODO: check whether it is needed
        "--res_id_label_shift",
        default=0,
        type=int,
        help="Shift residue ID labels by a specified amount in plots (default: 0).",
    )

    parser.add_argument(
        "--no_label_plots",
        action="store_true",
        help="Creates all the plots without labels as well.",
    )

    parser.add_argument(
        "--dont_save_graph_objects",
        action="store_true",
        help="Don't save the metadata and full graph objects of the calculations. Use this flag if there is not enough space for the calculation results or when the graph objects are not needed for further calculations or analysis.",
    )

    parser.add_argument(
        "--inter_monomer",
        default=True,
        action="store_true",
        help="Calculate and plot inter-monomer conserved interactions (default: True).",
    )

    parser.add_argument(
        "--collect_angles",
        default=False,
        action="store_true",
        help="Create a csv file with the angles of all donor-acceptor pairs that are within the set H-bond distance criterion in each frame (default: False).",
    )
    parser.add_argument(
        "--path",
        default=None,
        nargs=2,
        help="Search for paths between start and end nodes. Prove parameter in a form of --path start_node end_node.  Node name has to be given in a form of resname-resid e.g: ASP-213",
    )

    args = parser.parse_args()

    base = os.path.basename(args.psf)
    base_name, ext = os.path.splitext(base)

    dcd_files = []
    for dcd_file in args.dcd:
        dcd_files += glob.glob(dcd_file)
    dcd_files.sort()

    if not dcd_files:
        raise FileNotFoundError("No valid DCD files found.")

    output_folder = (
        args.output_folder if args.output_folder else os.path.dirname(args.psf)
    )
    os.makedirs(output_folder, exist_ok=True)

    if args.wrap_dcd:
        wrap_dcd = args.wrap_dcd.lower() == "true"
    else:
        wrap_dcd = True

    path = tuple(args.path) if args.path else None
    if path and args.root:
        raise ValueError("Connected component and path search can not be executed in the same computation.")

    oligo_comp = OligoComp(
        target_folder=output_folder,
        psf_file=args.psf,
        dcd_files=dcd_files,
        segment_names=args.segment_names,
        plot_parameters=ast.literal_eval(args.plot_parameters),
        dont_save_graph_objects=args.dont_save_graph_objects,
    )
    oligo_comp.calculate_graphs(
        max_water=int(args.max_water),
        check_angle=True,
        selection=args.selection,
        additional_donors=ast.literal_eval(args.additional_donors),
        additional_acceptors=ast.literal_eval(args.additional_acceptors),
        residuewise=args.residuewise,
        distance=args.distance,
        cut_angle=args.cut_angle,
        wrap_dcd=wrap_dcd,
        step=args.step,
        start=args.start,
        stop=args.stop,
        include_backbone_sidechain=args.include_backbone,
        connected_component_root=args.root,
        path=path,
        occupancy=float(args.occupancy),
        dont_save_graph_objects=args.dont_save_graph_objects,
        collect_angles=args.collect_angles
    )
    oligo_comp.plot_graphs(
        label_nodes=True,
        occupancy=float(args.occupancy),
    )

    oligo_comp.plot_conserved_graph(
        label_nodes=True,
        conservation_threshold=float(args.conservation_threshold),
        occupancy=float(args.occupancy)
    )
    oligo_comp.plot_differnece_graphs(
        label_nodes=True,
        conservation_threshold=float(args.conservation_threshold),
        occupancy=float(args.occupancy))

    if args.no_label_plots:
        oligo_comp.plot_graphs(
            label_nodes=False,
            label_edges=False,
            occupancy=float(args.occupancy),
        )

        oligo_comp.plot_conserved_graph(
            label_nodes=False,
            label_edges=False,
            conservation_threshold=float(args.conservation_threshold),
            occupancy=float(args.occupancy)
        )
        oligo_comp.plot_differnece_graphs(
            label_nodes=False,
            label_edges=False,
            conservation_threshold=float(args.conservation_threshold),
            occupancy=float(args.occupancy))

    if args.inter_monomer:
        oligo_comp_inter = OligoComp(
            target_folder=output_folder,
            psf_file=args.psf,
            dcd_files=dcd_files,
            segment_names=['inter_monomer'],
            plot_parameters=ast.literal_eval(args.plot_parameters),
            dont_save_graph_objects=args.dont_save_graph_objects,
        )

        oligo_comp_inter.calculate_graphs(
            max_water=int(args.max_water),
            check_angle=True,
            selection=args.selection,
            additional_donors=ast.literal_eval(args.additional_donors),
            additional_acceptors=ast.literal_eval(args.additional_acceptors),
            residuewise=args.residuewise,
            distance=args.distance,
            cut_angle=args.cut_angle,
            wrap_dcd=wrap_dcd,
            step=args.step,
            start=args.start,
            stop=args.stop,
            include_backbone_sidechain=args.include_backbone,
            connected_component_root=args.root,
            path=path,
            occupancy=float(args.occupancy),
            dont_save_graph_objects=args.dont_save_graph_objects,
            inter_monomer=args.inter_monomer
        )
        oligo_comp_inter.plot_graphs(
            label_nodes=True,
            occupancy=float(args.occupancy),
            inter_monomer=False
        )
        oligo_comp_inter.plot_graphs(
            label_nodes=True,
            occupancy=float(args.occupancy),
            inter_monomer=True
        )

        if args.no_label_plots:
            oligo_comp_inter.plot_graphs(
                label_nodes=False,
                occupancy=float(args.occupancy),
                inter_monomer=False
            )
            oligo_comp_inter.plot_graphs(
                label_nodes=False,
                occupancy=float(args.occupancy),
                inter_monomer=True
            )


if __name__ == "__main__":
    main()
