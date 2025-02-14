#!/usr/bin/env python
"""
Enhanced Linked Data Network Visualization Application
Incorporates improvements in UI/UX, including a tabbed interface, grouped sidebar controls,
dark mode enhancements, improved SPARQL feedback, and better graph editing.
Author: Huw Sandaver (Original) | Enhancements by ChatGPT
Date: 2025-02-14
"""

import os
import json
import logging
import traceback
import time
from io import StringIO
from urllib.parse import urlparse, urlunparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

import requests
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from rdflib import Graph as RDFGraph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS
import networkx as nx
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------
@dataclass
class Edge:
    source: str
    target: str
    relationship: str

@dataclass
class Node:
    id: str
    label: str
    types: List[str]
    metadata: Any  # metadata might not always be a dict
    edges: List[Edge] = field(default_factory=list)

@dataclass
class GraphData:
    nodes: List[Node]

# -----------------------------------------------------------------------------
# Configuration Constants (Consider moving these to a separate config.py for maintainability)
# -----------------------------------------------------------------------------
RELATIONSHIP_CONFIG: Dict[str, str] = {
    "relatedPerson": "#7289DA",
    "influencedBy": "#9B59B6",
    "memberOf": "#2ECC71",
    "succeededBy": "#E74C3C",
    "precededBy": "#FF69B4",
    "relatedOrganization": "#F1C40F",
    "headquarters": "#16A085",
    "administrativelyPartOf": "#3498DB",
    "placeOfBirth": "#FF7F50",
    "placeOfDeath": "#C0392B",
    "residence": "#8A2BE2",
    "draftsman": "#B22222",
    "depicts": "#FF8C00",
    "relatedWork": "#20B2AA",
    "creator": "#FF1493",
    "contributor": "#98FB98",
    "associatedWith": "#DDA0DD",
    "sameAs": "#A0522D",
    "child": "#1E90FF",
    "sibling": "#556B2F",
    "spouse": "#CD853F",
    "studentOf": "#8B008B",
    "employedBy": "#B8860B",
    "occupation": "#8FBC8F",
    "fieldOfActivity": "#FF4500",
    "educatedAt": "#8B4513",
    "foundedBy": "#FF6347",
    "containedInPlace": "#F39C12",
    "owner": "#3498DB"
}

NODE_TYPE_COLORS: Dict[str, str] = {
    "Person": "#FFA500",
    "Organization": "#87CEEB",
    "Place": "#98FB98",
    "StillImage": "#FFD700",
    "Event": "#DDA0DD",
    "Work": "#20B2AA",
    "Unknown": "#D3D3D3"
}

NODE_TYPE_SHAPES: Dict[str, str] = {
    "Person": "circle",
    "Organization": "box",
    "Place": "triangle",
    "StillImage": "dot",
    "Event": "star",
    "Work": "ellipse",
    "Unknown": "dot"
}

DEFAULT_NODE_COLOR = "#D3D3D3"
EX = Namespace("http://example.org/")  # Custom namespace for RDF

# -----------------------------------------------------------------------------
# Performance Monitoring Decorator
# -----------------------------------------------------------------------------
def performance_monitor(func):
    """Decorator to log the performance of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logging.info(f"[PERF] {func.__name__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def log_error(message: str) -> None:
    """Log error messages."""
    logging.error(message)

def remove_fragment(uri: str) -> str:
    """Remove the fragment (e.g., #something) from a URI."""
    try:
        parsed = urlparse(uri)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ''))
    except Exception as e:
        log_error(f"Error removing fragment from {uri}: {e}")
        return uri

def normalize_relationship_value(rel: str, value: Any) -> Optional[str]:
    if isinstance(value, dict):
        if rel in {"spouse", "studentOf", "employedBy", "educatedAt", "contributor", "draftsman", "creator", "owner"}:
            return remove_fragment(value.get('carriedOutBy', value.get('id', '')))
        elif rel == 'succeededBy':
            return remove_fragment(value.get('resultedIn', ''))
        elif rel == 'precededBy':
            return remove_fragment(value.get('resultedFrom', ''))
        elif rel == "foundedBy":
            return remove_fragment(value.get('carriedOutBy', value.get('founder', value.get('id', ''))))
        else:
            return remove_fragment(value.get('id', ''))
    elif isinstance(value, str):
        return remove_fragment(value)
    return None

def normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a JSON entity:
      - Remove fragment from 'id'.
      - Ensure 'prefLabel.en' exists.
      - Convert 'type' to a list.
      - Normalize relationships based on RELATIONSHIP_CONFIG.
    """
    if not isinstance(data, dict):
        raise ValueError("Invalid data format. Expected a dictionary.")

    data['id'] = remove_fragment(data.get('id', ''))
    data.setdefault('prefLabel', {})['en'] = data.get('prefLabel', {}).get('en', data['id'])

    if 'type' in data:
        data['type'] = data['type'] if isinstance(data['type'], list) else [data['type']]

    for rel in list(data.keys()):
        if rel not in RELATIONSHIP_CONFIG:
            continue
        values = data[rel]
        normalized_values = []
        if not isinstance(values, list):
            values = [values]
        for value in values:
            normalized_id = normalize_relationship_value(rel, value)
            if normalized_id:
                normalized_values.append(normalized_id)
                logging.debug(f"Normalized relationship '{rel}': {data['id']} -> {normalized_id}")
        data[rel] = normalized_values
    return data

# -----------------------------------------------------------------------------
# Data Caching: Parse Entities from File Contents
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
@performance_monitor
def parse_entities_from_contents(file_contents: List[str]) -> Tuple[GraphData, Dict[str, str], List[str]]:
    """
    Parse multiple JSON file contents into normalized entities using data classes.
    Returns a tuple: (GraphData, id_to_label, errors)
    """
    nodes: List[Node] = []
    id_to_label: Dict[str, str] = {}
    errors: List[str] = []

    for idx, content in enumerate(file_contents):
        try:
            json_obj = json.loads(content)
            normalized = normalize_data(json_obj)
            subject_id: str = normalized['id']
            label: str = normalized['prefLabel']['en']
            entity_types: List[str] = normalized.get('type', ['Unknown'])
            edges: List[Edge] = []
            for rel in RELATIONSHIP_CONFIG:
                values = normalized.get(rel, [])
                if not isinstance(values, list):
                    values = [values]
                for value in values:
                    normalized_id = normalize_relationship_value(rel, value)
                    if normalized_id:
                        edges.append(Edge(source=subject_id, target=normalized_id, relationship=rel))
            new_node = Node(id=subject_id, label=label, types=entity_types, metadata=normalized, edges=edges)
            nodes.append(new_node)
            id_to_label[subject_id] = label
        except Exception as e:
            err = f"File {idx}: {str(e)}\n{traceback.format_exc()}"
            errors.append(err)
            log_error(err)
    return GraphData(nodes=nodes), id_to_label, errors

# -----------------------------------------------------------------------------
# RDF Conversion and SPARQL Querying Helpers
# -----------------------------------------------------------------------------
def convert_graph_data_to_rdf(graph_data: GraphData) -> RDFGraph:
    """
    Convert GraphData (parsed JSON entities) to an RDF graph.
    Each entity becomes a subject and triples are added for labels, types, and relationships.
    """
    g = RDFGraph()
    g.bind("ex", EX)
    for node in graph_data.nodes:
        subject = URIRef(node.id)
        label = node.metadata.get("prefLabel", {}).get("en", node.id)
        g.add((subject, RDFS.label, Literal(label, lang="en")))
        for t in node.types:
            g.add((subject, RDF.type, EX[t]))
        for key, value in node.metadata.items():
            if key in ("id", "prefLabel", "type"):
                continue
            if key in RELATIONSHIP_CONFIG:
                if not isinstance(value, list):
                    value = [value]
                for v in value:
                    g.add((subject, EX[key], URIRef(v)))
            else:
                if isinstance(value, str):
                    g.add((subject, EX[key], Literal(value)))
                elif isinstance(value, list):
                    for v in value:
                        g.add((subject, EX[key], Literal(v)))
                else:
                    g.add((subject, EX[key], Literal(value)))
        for edge in node.edges:
            g.add((subject, EX[edge.relationship], URIRef(edge.target)))
    return g

def run_sparql_query(query: str, rdf_graph: RDFGraph) -> Set[str]:
    """
    Run a SPARQL SELECT query against the RDF graph.
    Returns a set of subject URIs (as strings) from the first variable.
    """
    result = rdf_graph.query(query, initNs={'rdf': RDF, 'ex': EX})
    nodes = {str(row[0]) for row in result if row[0] is not None}
    return nodes

# -----------------------------------------------------------------------------
# Graph Building Helpers
# -----------------------------------------------------------------------------
def add_node(net: Network, node_id: str, label: str, entity_types: List[str], color: str, metadata: Dict[str, Any],
             search_node: Optional[str] = None, show_labels: bool = True) -> None:
    """
    Add a node to the Pyvis network with enhanced aesthetics.
    """
    node_title = f"{label}\nTypes: {', '.join(entity_types)}"
    description = ""
    if "description" in metadata:
        if isinstance(metadata["description"], dict):
            description = metadata["description"].get("en", "")
        elif isinstance(metadata["description"], str):
            description = metadata["description"]
    if description:
        node_title += f"\nDescription: {description}"
    
    net.add_node(
        node_id,
        label=label if show_labels else "",
        title=node_title,
        color=color,
        shape=NODE_TYPE_SHAPES.get(entity_types[0], "dot") if entity_types else "dot",
        size=20 if (search_node is not None and node_id == search_node) else 15,
        font={"size": 12, "face": "Arial", "color": "#343a40"},
        borderWidth=2 if (search_node is not None and node_id == search_node) else 1,
        borderColor="#FF5733" if (search_node is not None and node_id == search_node) else "#343a40",
        shadow=True,
        widthConstraint={"maximum": 150}
    )
    logging.debug(f"Added node: {label} ({node_id}) with color {color}")

def add_edge(net: Network, src: str, dst: str, relationship: str, id_to_label: Dict[str, str],
             search_node: Optional[str] = None) -> None:
    """Add an edge to the Pyvis network with enhanced styling."""
    edge_color = RELATIONSHIP_CONFIG.get(relationship, "#A9A9A9")
    label_text = " ".join(word.capitalize() for word in relationship.split('_'))
    is_search_edge = search_node is not None and (src == search_node or dst == search_node)
    net.add_edge(
        src,
        dst,
        label=label_text,
        color="#FF5733" if is_search_edge else edge_color,
        width=3 if is_search_edge else 2,
        arrows='to',
        title=f"{label_text}: {id_to_label.get(src, src)} → {id_to_label.get(dst, dst)}",
        font={"size": 10, "align": "middle"},
        smooth={'enabled': True, 'type': 'continuous'}
    )
    logging.debug(f"Added edge: {src} --{label_text}--> {dst}")

def build_graph(
    graph_data: GraphData,
    id_to_label: Dict[str, str],
    selected_relationships: List[str],
    search_node: Optional[str] = None,
    node_positions: Optional[Dict[str, Dict[str, float]]] = None,
    show_labels: bool = True,
    filtered_nodes: Optional[Set[str]] = None,
    dark_mode: bool = False
) -> Network:
    """
    Build and configure the Pyvis graph based on the parsed data.
    """
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#f0f2f6" if not dark_mode else "#2c2f33",
        font_color="#343a40" if not dark_mode else "#ffffff"
    )
    # Apply physics settings
    net.force_atlas_2based(
        gravity=st.session_state.physics_params.get("gravity", -50),
        central_gravity=st.session_state.physics_params.get("centralGravity", 0.01),
        spring_length=st.session_state.physics_params.get("springLength", 150),
        spring_strength=st.session_state.physics_params.get("springStrength", 0.08)
    )

    added_nodes: Set[str] = set()
    edge_set: Set[Tuple[str, str, str]] = set()

    for node in graph_data.nodes:
        if filtered_nodes is not None and node.id not in filtered_nodes:
            logging.debug(f"Skipping node {node.id} due to filtering")
            continue
        color = next((NODE_TYPE_COLORS.get(t, DEFAULT_NODE_COLOR) for t in node.types), DEFAULT_NODE_COLOR)
        if node.id not in added_nodes:
            add_node(net, node.id, id_to_label.get(node.id, node.id), node.types, color, node.metadata,
                     search_node=search_node, show_labels=show_labels)
            added_nodes.add(node.id)

    for node in graph_data.nodes:
        for edge in node.edges:
            if edge.relationship not in selected_relationships:
                continue
            if filtered_nodes is not None and (edge.source not in filtered_nodes or edge.target not in filtered_nodes):
                logging.debug(f"Skipping edge {edge.source} --{edge.relationship}--> {edge.target} due to filtering")
                continue
            if edge.target not in added_nodes:
                target_label = id_to_label.get(edge.target, edge.target)
                add_node(net, edge.target, target_label, ["Unknown"], DEFAULT_NODE_COLOR, {},
                         show_labels=show_labels)
                added_nodes.add(edge.target)
            if (edge.source, edge.target, edge.relationship) not in edge_set:
                add_edge(net, edge.source, edge.target, edge.relationship, id_to_label, search_node=search_node)
                edge_set.add((edge.source, edge.target, edge.relationship))

    node_count = len(net.nodes)
    node_font_size = 12 if node_count <= 50 else 10
    edge_font_size = 10 if node_count <= 50 else 8

    net.options = json.loads(f"""
    {{
        "nodes": {{
            "font": {{
                "size": {node_font_size},
                "face": "Arial",
                "color": "{'#343a40' if not dark_mode else '#ffffff'}",
                "strokeWidth": 0
            }}
        }},
        "edges": {{
            "font": {{
                "size": {edge_font_size},
                "face": "Arial",
                "align": "middle",
                "color": "{'#343a40' if not dark_mode else '#ffffff'}"
            }},
            "smooth": {{
                "type": "continuous"
            }}
        }},
        "physics": {{
            "forceAtlas2Based": {{
                "gravity": {st.session_state.physics_params.get("gravity", -50)},
                "centralGravity": {st.session_state.physics_params.get("centralGravity", 0.01)},
                "springLength": {st.session_state.physics_params.get("springLength", 150)},
                "springStrength": {st.session_state.physics_params.get("springStrength", 0.08)}
            }},
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        }},
        "interaction": {{
            "hover": true,
            "navigationButtons": true,
            "zoomView": true,
            "dragNodes": true,
            "multiselect": true,
            "selectConnectedEdges": true
        }}
    }}
    """)
    return net

def create_legends(relationship_colors: Dict[str, str], node_type_colors: Dict[str, str]) -> str:
    """Generate an HTML legend for relationship types and node types."""
    relationship_items = "".join(
        f"<li><span style='color:{color}; font-size: 16px;'>●</span> {rel.replace('_', ' ').title()}</li>"
        for rel, color in relationship_colors.items()
    )
    node_type_items = "".join(
        f"<li><span style='color:{color}; font-size: 16px;'>●</span> {ntype}</li>"
        for ntype, color in node_type_colors.items()
    )
    return (
        f"<h4>Legends</h4>"
        f"<div style='display:flex;'>"
        f"<ul style='list-style: none; padding: 0; margin-right: 20px;'>"
        f"<strong>Relationships</strong>{relationship_items}"
        f"</ul>"
        f"<ul style='list-style: none; padding: 0;'>"
        f"<strong>Node Types</strong>{node_type_items}"
        f"</ul>"
        f"</div>"
    )

def display_node_metadata(node_id: str, graph_data: GraphData, id_to_label: Dict[str, str]) -> None:
    """Display metadata for a given node with formatted markdown."""
    st.markdown("#### Node Metadata")
    node_obj = next((node for node in graph_data.nodes if node.id == node_id), None)
    if node_obj:
        st.markdown(f"**Label:** {id_to_label.get(node_obj.id, node_obj.id)}")
        if isinstance(node_obj.metadata, dict):
            for key, value in node_obj.metadata.items():
                if key == 'prefLabel':
                    continue
                st.markdown(f"**{key}:** {value}")
        else:
            st.write("Metadata:", node_obj.metadata)
    else:
        st.write("No metadata available for this node.")

# -----------------------------------------------------------------------------
# Main Streamlit App
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Linked Data Explorer", page_icon="🕸️", layout="wide")
    
    # Initialize session state variables if not already set
    if "node_positions" not in st.session_state:
        st.session_state.node_positions = {}
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = None
    if "selected_relationships" not in st.session_state:
        st.session_state.selected_relationships = list(RELATIONSHIP_CONFIG.keys())
    if "search_term" not in st.session_state:
        st.session_state.search_term = ""
    if "show_labels" not in st.session_state:
        st.session_state.show_labels = True
    if "sparql_query" not in st.session_state:
        st.session_state.sparql_query = ""
    if "filtered_types" not in st.session_state:
        st.session_state.filtered_types = []
    if "enable_physics" not in st.session_state:
        st.session_state.enable_physics = True
    if "graph_data" not in st.session_state:
        st.session_state.graph_data = GraphData(nodes=[])
    if "id_to_label" not in st.session_state:
        st.session_state.id_to_label = {}
    if "physics_params" not in st.session_state:
        st.session_state.physics_params = {
            "gravity": -50,
            "centralGravity": 0.01,
            "springLength": 150,
            "springStrength": 0.08
        }
        
    # -----------------------------------------------------------------------------
    # Sidebar: Grouped Controls with Expanders
    # -----------------------------------------------------------------------------
    st.sidebar.header("Controls")
    
    with st.sidebar.expander("File Upload"):
        uploaded_files = st.sidebar.file_uploader(
            label="Upload JSON Files",
            type=["json"],
            accept_multiple_files=True,
            help="Select JSON files describing entities and relationships"
        )
        if uploaded_files:
            file_contents = [uploaded_file.read().decode("utf-8") for uploaded_file in uploaded_files]
            graph_data, id_to_label, errors = parse_entities_from_contents(file_contents)
            st.session_state.graph_data = graph_data
            st.session_state.id_to_label = id_to_label
            st.session_state.rdf_graph = convert_graph_data_to_rdf(graph_data)
            if errors:
                for error in errors:
                    st.error(error)
    
    with st.sidebar.expander("Appearance & Settings"):
        dark_mode = st.checkbox("Dark Mode", value=False, key="dark_mode")
        community_detection = st.checkbox("Enable Community Detection", value=False, key="community_detection")
    
    with st.sidebar.expander("Physics Settings"):
        st.session_state.physics_params["gravity"] = st.number_input(
            "Gravity",
            value=float(st.session_state.physics_params["gravity"]),
            step=1.0,
            key="gravity_input"
        )
        st.session_state.physics_params["centralGravity"] = st.number_input(
            "Central Gravity",
            value=st.session_state.physics_params["centralGravity"],
            step=0.01,
            key="centralGravity_input"
        )
        st.session_state.physics_params["springLength"] = st.number_input(
            "Spring Length",
            value=float(st.session_state.physics_params["springLength"]),
            step=1.0,
            key="springLength_input"
        )
        st.session_state.physics_params["springStrength"] = st.number_input(
            "Spring Strength",
            value=st.session_state.physics_params["springStrength"],
            step=0.01,
            key="springStrength_input"
        )
    
    with st.sidebar.expander("Graph Settings"):
        graph_settings_file = st.file_uploader("Upload Graph Settings", type=["json"], key="graph_settings_file")
        if graph_settings_file is not None:
            try:
                settings_str = graph_settings_file.read().decode("utf-8")
                settings_json = json.loads(settings_str)
                st.session_state.node_positions = settings_json.get("node_positions", st.session_state.node_positions)
                st.session_state.selected_relationships = settings_json.get("selected_relationships", st.session_state.selected_relationships)
                st.session_state.enable_physics = settings_json.get("enable_physics", st.session_state.enable_physics)
                st.session_state.filtered_types = settings_json.get("filtered_types", st.session_state.filtered_types)
                st.session_state.search_term = settings_json.get("search_term", st.session_state.search_term)
                st.session_state.show_labels = settings_json.get("show_labels", st.session_state.show_labels)
                st.session_state.physics_params = settings_json.get("physics_params", st.session_state.physics_params)
                st.sidebar.success("Graph settings uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading graph settings: {e}")
        graph_settings = {
            "node_positions": st.session_state.node_positions,
            "selected_relationships": st.session_state.selected_relationships,
            "enable_physics": st.session_state.enable_physics,
            "filtered_types": st.session_state.filtered_types,
            "search_term": st.session_state.search_term,
            "show_labels": st.session_state.show_labels,
            "physics_params": st.session_state.physics_params,
        }
        settings_json_str = json.dumps(graph_settings, indent=2)
        st.download_button(
            label="Download Graph Settings",
            data=settings_json_str,
            file_name="graph_settings.json",
            mime="application/json"
        )
    
    with st.sidebar.expander("Graph Data Loader"):
        graph_data_file = st.file_uploader("Upload Graph Data JSON", type=["json"], key="graph_data_file")
        if graph_data_file is not None:
            try:
                graph_data_str = graph_data_file.read().decode("utf-8")
                graph_data_json = json.loads(graph_data_str)
                nodes = []
                id_to_label = {}
                for node_dict in graph_data_json.get("nodes", []):
                    node_obj = Node(
                        id=node_dict.get("id"),
                        label=node_dict.get("label"),
                        types=node_dict.get("types", ["Unknown"]),
                        metadata=node_dict.get("metadata", {}),
                        edges=[]
                    )
                    nodes.append(node_obj)
                    id_to_label[node_obj.id] = node_obj.label
                for edge_dict in graph_data_json.get("edges", []):
                    src = edge_dict.get("from")
                    dst = edge_dict.get("to")
                    relationship = edge_dict.get("label", "related")
                    new_edge = Edge(source=src, target=dst, relationship=relationship)
                    for node in nodes:
                        if node.id == src:
                            node.edges.append(new_edge)
                            break
                st.session_state.graph_data = GraphData(nodes=nodes)
                st.session_state.id_to_label = id_to_label
                st.sidebar.success("Graph data loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading graph data: {e}")
    
    with st.sidebar.expander("Graph Editing"):
        edit_option = st.radio("Select action", ("Add Node", "Delete Node", "Modify Node", "Add Edge", "Delete Edge"))
        nodes_list: List[Node] = st.session_state.graph_data.nodes
        if edit_option == "Add Node":
            with st.form("add_node_form"):
                new_node_id = st.text_input("Node ID")
                new_node_label = st.text_input("Node Label")
                new_node_type = st.selectbox("Node Type", list(NODE_TYPE_COLORS.keys()))
                submitted = st.form_submit_button("Add Node")
                if submitted:
                    if any(node.id == new_node_id for node in nodes_list):
                        st.error("A node with this ID already exists.")
                    elif new_node_id and new_node_label:
                        new_node = Node(
                            id=new_node_id,
                            label=new_node_label,
                            types=[new_node_type],
                            metadata={
                                "id": new_node_id,
                                "prefLabel": {"en": new_node_label},
                                "type": [new_node_type]
                            },
                            edges=[]
                        )
                        nodes_list.append(new_node)
                        st.session_state.id_to_label[new_node_id] = new_node_label
                        st.sidebar.success(f"Node '{new_node_label}' added.")
                    else:
                        st.error("Please provide both a Node ID and Label.")
        elif edit_option == "Delete Node":
            node_ids = [node.id for node in nodes_list]
            if node_ids:
                default_index = node_ids.index(st.session_state.selected_node) if st.session_state.selected_node in node_ids else 0
                node_to_delete = st.selectbox("Select Node to Delete", node_ids, index=default_index)
                if st.button("Delete Node"):
                    st.session_state.graph_data.nodes = [node for node in nodes_list if node.id != node_to_delete]
                    for node in st.session_state.graph_data.nodes:
                        node.edges = [edge for edge in node.edges if edge.target != node_to_delete]
                    st.session_state.id_to_label.pop(node_to_delete, None)
                    st.sidebar.success(f"Node '{node_to_delete}' deleted.")
            else:
                st.info("No nodes available to delete.")
        elif edit_option == "Modify Node":
            node_ids = [node.id for node in nodes_list]
            if node_ids:
                default_index = node_ids.index(st.session_state.selected_node) if st.session_state.selected_node in node_ids else 0
                node_to_modify = st.selectbox("Select Node to Modify", node_ids, index=default_index)
                node_obj = next((node for node in nodes_list if node.id == node_to_modify), None)
                if node_obj:
                    with st.form("modify_node_form"):
                        new_label = st.text_input("New Label", value=node_obj.label)
                        new_type = st.selectbox("New Type", list(NODE_TYPE_COLORS.keys()),
                                                index=list(NODE_TYPE_COLORS.keys()).index(node_obj.types[0])
                                                if node_obj.types[0] in NODE_TYPE_COLORS else 0)
                        submitted = st.form_submit_button("Modify Node")
                        if submitted:
                            node_obj.label = new_label
                            node_obj.types = [new_type]
                            node_obj.metadata["prefLabel"]["en"] = new_label
                            st.session_state.id_to_label[node_to_modify] = new_label
                            st.sidebar.success(f"Node '{node_to_modify}' modified.")
            else:
                st.info("No nodes available to modify.")
        elif edit_option == "Add Edge":
            if nodes_list:
                with st.form("add_edge_form"):
                    source_node = st.selectbox("Source Node", [node.id for node in nodes_list])
                    target_node = st.selectbox("Target Node", [node.id for node in nodes_list])
                    relationship = st.selectbox("Relationship", list(RELATIONSHIP_CONFIG.keys()))
                    submitted = st.form_submit_button("Add Edge")
                    if submitted:
                        for node in nodes_list:
                            if node.id == source_node:
                                node.edges.append(Edge(source=source_node, target=target_node, relationship=relationship))
                        st.sidebar.success(f"Edge '{relationship}' from '{source_node}' to '{target_node}' added.")
            else:
                st.info("No nodes available to add an edge.")
        elif edit_option == "Delete Edge":
            all_edges = []
            for node in nodes_list:
                for edge in node.edges:
                    all_edges.append((edge.source, edge.target, edge.relationship))
            if all_edges:
                edge_to_delete = st.selectbox("Select Edge to Delete", all_edges)
                if st.button("Delete Edge"):
                    for node in nodes_list:
                        if node.id == edge_to_delete[0]:
                            node.edges = [edge for edge in node.edges if (edge.source, edge.target, edge.relationship) != edge_to_delete]
                    st.sidebar.success("Edge deleted.")
            else:
                st.info("No edges available to delete.")
    
    with st.sidebar.expander("Manual Node Positioning"):
        if st.session_state.graph_data.nodes:
            unique_nodes: Dict[str, str] = {node.id: node.label for node in st.session_state.graph_data.nodes}
            for node in st.session_state.graph_data.nodes:
                for edge in node.edges:
                    unique_nodes.setdefault(edge.target, st.session_state.id_to_label.get(edge.target, edge.target))
            node_ids = list(unique_nodes.keys())
            default_index = node_ids.index(st.session_state.selected_node) if st.session_state.selected_node in node_ids else 0
            selected_node = st.selectbox(
                label="Select a Node to Position",
                options=node_ids,
                index=default_index,
                format_func=lambda x: unique_nodes.get(x, x),
                key="selected_node_control"
            )
            st.session_state.selected_node = selected_node
            if selected_node:
                with st.form(key="position_form"):
                    x_pos = st.number_input(
                        "X Position",
                        value=st.session_state.node_positions.get(selected_node, {}).get("x", 0.0),
                        step=10.0,
                        help="Set the X coordinate for the selected node."
                    )
                    y_pos = st.number_input(
                        "Y Position",
                        value=st.session_state.node_positions.get(selected_node, {}).get("y", 0.0),
                        step=10.0,
                        help="Set the Y coordinate for the selected node."
                    )
                    if st.form_submit_button(label="Set Position"):
                        st.session_state.node_positions[selected_node] = {"x": x_pos, "y": y_pos}
                        st.sidebar.success(f"Position for '{unique_nodes[selected_node]}' set to (X: {x_pos}, Y: {y_pos})")
    
    # ------------------ Filter by Entity Types ------------------
    if st.session_state.graph_data.nodes:
        all_types = {t for node in st.session_state.graph_data.nodes for t in node.types}
        st.session_state.filtered_types = st.sidebar.multiselect(
            "Filter by Entity Types",
            options=list(all_types),
            default=st.session_state.filtered_types,
            key="filtered_types_control"
        )
    
    st.session_state.search_term = st.sidebar.text_input(
        label="Search for a Node",
        help="Enter the entity name to highlight",
        key="search_term_control",
        value=st.session_state.search_term
    )
    sparql_query_input = st.sidebar.text_area(
        label="SPARQL Query",
        help="Enter a SPARQL SELECT query to filter nodes. For example:\n```\nSELECT ?s WHERE { ?s rdf:type ex:Person . }\n```",
        key="sparql_query_control",
        value=st.session_state.sparql_query
    )
    st.session_state.sparql_query = sparql_query_input

    if sparql_query_input.strip():
        st.sidebar.info("Query Running...")
        try:
            filtered_nodes = run_sparql_query(sparql_query_input, st.session_state.rdf_graph)
            st.sidebar.success(f"Query Successful: {len(filtered_nodes)} result(s) found.")
            st.sidebar.dataframe(pd.DataFrame(list(filtered_nodes), columns=["Node ID"]))
        except Exception as e:
            st.sidebar.error(f"SPARQL Query failed: {e}")
            filtered_nodes = None
    else:
        filtered_nodes = None

    # <-- FIX: Apply filtering by selected entity types -->
    if st.session_state.filtered_types:
        filtered_by_type = {node.id for node in st.session_state.graph_data.nodes
                            if any(t in st.session_state.filtered_types for t in node.types)}
        filtered_nodes = filtered_nodes.intersection(filtered_by_type) if filtered_nodes is not None else filtered_by_type

    st.sidebar.markdown(create_legends(RELATIONSHIP_CONFIG, NODE_TYPE_COLORS), unsafe_allow_html=True)
    
    # -----------------------------------------------------------------------------
    # Main Content: Tabbed Interface
    # -----------------------------------------------------------------------------
    tabs = st.tabs(["Graph View", "Data View", "SPARQL Query", "About"])
    
    # ------------------- Tab 1: Graph View -------------------
    with tabs[0]:
        st.header("🕸️ Network Graph")
        if st.session_state.graph_data.nodes:
            with st.spinner("Generating Network Graph..."):
                net = build_graph(
                    graph_data=st.session_state.graph_data,
                    id_to_label=st.session_state.id_to_label,
                    selected_relationships=st.session_state.selected_relationships,
                    search_node=next((node.id for node in st.session_state.graph_data.nodes 
                                      if node.label.lower() == st.session_state.search_term.lower()), None),
                    node_positions=st.session_state.node_positions,
                    show_labels=st.session_state.show_labels,
                    filtered_nodes=filtered_nodes,
                    dark_mode=dark_mode
                )
            # Apply manual node positions and fix them if physics is disabled
            if st.session_state.enable_physics:
                for node in net.nodes:
                    pos = st.session_state.node_positions.get(node.get("id"))
                    if pos:
                        node['x'] = pos['x']
                        node['y'] = pos['y']
                        node['fixed'] = True
            else:
                # Disable physics and use spring layout
                if isinstance(net.options, dict):
                    net.options["physics"]["enabled"] = False
                G = nx.Graph()
                for node in net.nodes:
                    G.add_node(node["id"])
                for edge in net.edges:
                    G.add_edge(edge["from"], edge["to"])
                iterations = 20 if len(G.nodes) <= 50 else 50
                pos = nx.spring_layout(G, k=0.15, iterations=iterations, seed=42)
                for node in net.nodes:
                    node_id = node["id"]
                    if st.session_state.node_positions.get(node_id):
                        node['x'] = st.session_state.node_positions[node_id]["x"]
                        node['y'] = st.session_state.node_positions[node_id]["y"]
                    else:
                        p = pos[node_id]
                        node['x'] = p[0] * 500
                        node['y'] = p[1] * 500
                    node['fixed'] = True
                    node['physics'] = False
            # Community detection enhancement
            if community_detection:
                G = nx.Graph()
                for node in net.nodes:
                    G.add_node(node["id"])
                for edge in net.edges:
                    G.add_edge(edge["from"], edge["to"])
                try:
                    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
                    community_colors = [
                        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
                        "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"
                    ]
                    community_map: Dict[str, str] = {}
                    for i, comm in enumerate(communities):
                        for n in comm:
                            community_map[n] = community_colors[i % len(community_colors)]
                    for node in net.nodes:
                        if node["id"] in community_map:
                            node["color"] = community_map[node["id"]]
                    if isinstance(net.options, dict):
                        net.options.setdefault("configure", {})["enabled"] = True
                        net.options["configure"]["filter"] = ["physics"]
                except Exception as e:
                    st.error(f"Community detection failed: {e}")
            if isinstance(net.options, dict):
                net.options["configure"] = {
                    "enabled": True,
                    "filter": ["physics"],
                    "showButton": True
                }
            if len(net.nodes) > 50 and st.session_state.show_labels:
                st.info("The graph has many nodes. Consider toggling 'Show Node Labels' off for better readability.")
            try:
                output_path = "network_graph.html"
                net.save_graph(output_path)
                with open(output_path, "r", encoding="utf-8") as f:
                    graph_html = f.read()
                os.remove(output_path)
                components.html(graph_html, height=750, scrolling=True)
            except Exception as e:
                st.error(f"Graph generation failed: {e}")
            st.markdown(f"**Total Nodes:** {len(net.nodes)} | **Total Edges:** {len(net.edges)}")
            if st.session_state.selected_node:
                display_node_metadata(st.session_state.selected_node, st.session_state.graph_data, st.session_state.id_to_label)
            else:
                st.markdown("#### Node Metadata")
                st.info("Select a node from the sidebar to view its metadata.")
            # Map Visualization Enhancement
            place_locations = []
            for node in st.session_state.graph_data.nodes:
                if isinstance(node.metadata, dict) and "geographicCoordinates" in node.metadata:
                    coords = node.metadata["geographicCoordinates"]
                    if isinstance(coords, list):
                        coords = coords[0]
                    if isinstance(coords, str) and coords.startswith("Point(") and coords.endswith(")"):
                        coords = coords[6:-1].strip()
                        parts = coords.split()
                        if len(parts) == 2:
                            try:
                                lon, lat = map(float, parts)
                                place_locations.append({"lat": lat, "lon": lon, "label": node.label})
                            except ValueError:
                                logging.error(f"Invalid coordinates for node {node.id}: {coords}")
                elif isinstance(node.metadata, dict) and "latitude" in node.metadata and "longitude" in node.metadata:
                    lat = node.metadata.get("latitude")
                    lon = node.metadata.get("longitude")
                    if lat is not None and lon is not None:
                        try:
                            place_locations.append({"lat": float(lat), "lon": float(lon), "label": node.label})
                        except ValueError:
                            logging.error(f"Invalid numeric coordinates for node {node.id}: {lat}, {lon}")
            if place_locations:
                st.subheader("Map View of Entities with Coordinates")
                df_places = pd.DataFrame(place_locations)
                st.map(df_places)
            else:
                st.info("No entities with valid coordinates found for the map view.")
            # IIIF Viewer Enhancement
            st.subheader("IIIF Viewer")
            iiif_nodes = [
                node for node in st.session_state.graph_data.nodes
                if isinstance(node.metadata, dict) and ("image" in node.metadata or "manifest" in node.metadata)
            ]
            if iiif_nodes:
                selected_iiif = st.selectbox(
                    "Select an entity with a manifest for IIIF Viewer",
                    options=[node.id for node in iiif_nodes],
                    format_func=lambda x: st.session_state.id_to_label.get(x, x)
                )
                selected_node_obj = next((node for node in iiif_nodes if node.id == selected_iiif), None)
                if selected_node_obj:
                    # Try 'image' first, if not found, then 'manifest'
                    manifest_url = selected_node_obj.metadata.get("image") or selected_node_obj.metadata.get("manifest")
                    if manifest_url:
                        if isinstance(manifest_url, list):
                            manifest_url = manifest_url[0]
                        prefix = "https://divinity.contentdm.oclc.org/digital/custom/mirador3?manifest="
                        if manifest_url.startswith(prefix):
                            manifest_url = manifest_url[len(prefix):]
                        html_code = f'''
<html>
  <head>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/mirador/dist/css/mirador.min.css">
    <script src="https://unpkg.com/mirador/dist/mirador.min.js"></script>
  </head>
  <body>
    <div id="mirador-viewer" style="height: 600px;"></div>
    <script>
      Mirador.viewer({{
        id: 'mirador-viewer',
        windows: [{{ loadedManifest: '{manifest_url}' }}]
      }});
    </script>
  </body>
</html>
'''
                        components.html(html_code, height=650)
                    else:
                        st.info("No valid manifest found for the selected entity.")
            else:
                st.info("No entity with a manifest found.")
            # Export Options
            st.markdown("### 📥 Export Options")
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'graph_html' in locals():
                    st.download_button(
                        label="Download Graph as HTML",
                        data=graph_html,
                        file_name="network_graph.html",
                        mime="text/html"
                    )
                else:
                    st.info("Please generate the graph first to download.")
            with col2:
                if st.button("Download Graph as PNG"):
                    st.warning("Exporting as PNG is not supported directly. Please export as HTML and take a screenshot.")
            with col3:
                if st.button("Download Graph Data as JSON"):
                    graph_data_json = {
                        "nodes": [
                            {"id": node.get('id'), "label": node.get('label'), "types": node.get("types", ["Unknown"]),
                             "metadata": node.get('title', ''), "x": node.get('x'), "y": node.get('y')}
                            for node in net.nodes
                        ],
                        "edges": [
                            {"from": edge.get('from'), "to": edge.get('to'), "label": edge.get('label'),
                             "metadata": edge.get('title', '')}
                            for edge in net.edges
                        ]
                    }
                    json_str = json.dumps(graph_data_json, indent=2)
                    st.download_button(
                        label="Download Graph Data as JSON",
                        data=json_str,
                        file_name="graph_data.json",
                        mime="application/json"
                    )
        else:
            st.info("No valid data found. Please check your JSON files.")
    
    # ------------------- Tab 2: Data View -------------------
    with tabs[1]:
        st.header("Data View")
        st.subheader("Graph Nodes")
        if st.session_state.graph_data.nodes:
            data = [{"ID": node.id, "Label": node.label, "Types": ", ".join(node.types)} for node in st.session_state.graph_data.nodes]
            st.dataframe(pd.DataFrame(data))
        else:
            st.info("No data available. Please upload JSON files.")
    
    # ------------------- Tab 3: SPARQL Query -------------------
    with tabs[2]:
        st.header("SPARQL Query")
        st.markdown("Enter a SPARQL SELECT query in the sidebar and view the results here.")
        if sparql_query_input.strip():
            try:
                query_results = run_sparql_query(sparql_query_input, st.session_state.rdf_graph)
                st.success(f"Query returned {len(query_results)} result(s).")
                st.dataframe(pd.DataFrame(list(query_results), columns=["Node ID"]))
            except Exception as e:
                st.error(f"SPARQL Query failed: {e}")
        else:
            st.info("No query entered.")
    
    # ------------------- Tab 4: About -------------------
    with tabs[3]:
        st.header("About Linked Data Explorer")
        st.markdown(
            """
            ### Explore Relationships Between Entities
            Upload multiple JSON files representing entities and generate an interactive network.
            Use the sidebar to filter relationships, search for nodes, set manual positions,
            and edit the graph directly.
            
            **Features:**
            - **Tabbed Interface:** Separate views for the graph, raw data, SPARQL queries, and about information.
            - **Dynamic Graph Visualization:** Interactive network graph with manual node positioning and community detection.
            - **Dark Mode:** Toggle for a dark-themed interface with improved readability.
            - **SPARQL Query Support:** Run queries on your RDF-converted graph and preview results.
            - **IIIF Viewer:** For entities with a manifest, view the associated manifest directly in the app.
            
            **Usage:**
            - Upload JSON files via the sidebar.
            - Configure graph settings and physics parameters.
            - Edit the graph by adding, deleting, or modifying nodes and edges.
            - Explore your data using the various tabs.
            
            Enjoy exploring your linked data!
            """
        )
    
    # -----------------------------------------------------------------------------
    # Global Style Adjustments for Dark Mode and Layout
    # -----------------------------------------------------------------------------
    st.markdown(
        f"""
        <style>
            .stApp {{
                max-width: 1600px;
                padding: 1rem;
                background-color: {"#2c2f33" if dark_mode else "#f0f2f6"};
                color: {"#ffffff" if dark_mode else "#343a40"};
            }}
            h1, h3, h4 {{
                color: {"#ddd" if dark_mode else "#333"};
            }}
            .stButton > button, .stDownloadButton > button {{
                background-color: #5cb85c;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                cursor: pointer;
                border-radius: 0.25rem;
            }}
            .stButton > button:hover, .stDownloadButton > button:hover {{
                background-color: #4cae4c;
            }}
            @media (max-width: 768px) {{
                .stApp {{
                    padding: 0.5rem;
                }}
            }}
        </style>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
