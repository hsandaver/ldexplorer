#!/usr/bin/env python
"""
Enhanced Linked Data Network Visualization Application with Custom Default Physics Options

Default physics options (disabled by default):
{
  "physics": {
    "enabled": false,
    "hierarchicalRepulsion": {
      "centralGravity": 0,
      "springLength": 230,
      "nodeDistance": 210,
      "avoidOverlap": 1
    },
    "minVelocity": 0.75,
    "solver": "hierarchicalRepulsion"
  }
}

Author: ChatGPT (Enhanced)
Date: 2025-02-15
"""

import os
import json
import logging
import traceback
import time
from urllib.parse import urlparse, urlunparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from rdflib import Graph as RDFGraph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS
import networkx as nx
import pandas as pd

# -----------------------------------------------------------------------------
# Set page configuration (must be the very first Streamlit command)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Linked Data Explorer", page_icon="🕸️", layout="wide")

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Constants and Configurations
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
    "AdministrativeArea": "#FFB6C1",
    "Unknown": "#D3D3D3"
}

NODE_TYPE_SHAPES: Dict[str, str] = {
    "Person": "circle",
    "Organization": "box",
    "Place": "triangle",
    "StillImage": "dot",
    "Event": "star",
    "Work": "ellipse",
    "AdministrativeArea": "diamond",
    "Unknown": "dot"
}

DEFAULT_NODE_COLOR = "#D3D3D3"
EX = Namespace("http://example.org/")  # Custom RDF namespace

# -----------------------------------------------------------------------------
# Data Classes (Module: Data Structures)
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
# Utility Functions Module
# -----------------------------------------------------------------------------
def log_error(message: str) -> None:
    """Log an error message."""
    logging.error(message)

def remove_fragment(uri: str) -> str:
    """Remove URI fragment for consistency."""
    try:
        parsed = urlparse(uri)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ''))
    except Exception as e:
        log_error(f"Error removing fragment from {uri}: {e}")
        return uri

def normalize_relationship_value(rel: str, value: Any) -> Optional[str]:
    """Normalize relationship value based on its type."""
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
    """Normalize a data dictionary representing an entity."""
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

def is_valid_iiif_manifest(url: str) -> bool:
    """Check if a URL is a valid IIIF manifest."""
    if not url.startswith("http"):
        return False
    lower_url = url.lower()
    return "iiif" in lower_url and ("manifest" in lower_url or lower_url.endswith("manifest.json"))

# -----------------------------------------------------------------------------
# Data Parsing and Caching Module
# -----------------------------------------------------------------------------
@st.cache_data
def parse_entities_from_contents(file_contents: List[str]) -> Tuple[GraphData, Dict[str, str], List[str]]:
    """
    Parse uploaded JSON contents into GraphData.
    
    Returns a tuple containing GraphData, a mapping from node IDs to labels, and a list of errors.
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
# RDF and SPARQL Helpers Module
# -----------------------------------------------------------------------------
def convert_graph_data_to_rdf(graph_data: GraphData) -> RDFGraph:
    """Convert GraphData into an RDF graph."""
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
    """Run a SPARQL SELECT query on the provided RDF graph."""
    result = rdf_graph.query(query, initNs={'rdf': RDF, 'ex': EX})
    return {str(row[0]) for row in result if row[0] is not None}

# -----------------------------------------------------------------------------
# Graph Building Helpers Module
# -----------------------------------------------------------------------------
def add_node(net: Network, node_id: str, label: str, entity_types: List[str],
             color: str, metadata: Dict[str, Any],
             search_node: Optional[str] = None, show_labels: bool = True) -> None:
    """
    Add a node to the pyvis network with dynamic styling based on density.
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
    
    # Dynamic sizing: highlight search node or adjust size based on density
    size = 20 if (search_node is not None and node_id == search_node) else 15

    net.add_node(
        node_id,
        label=label if show_labels else "",
        title=node_title,
        color=color,
        shape=NODE_TYPE_SHAPES.get(entity_types[0], "dot") if entity_types else "dot",
        size=size,
        font={"size": 12 if size == 20 else 10, "face": "Arial", "color": "#343a40"},
        borderWidth=2 if (search_node is not None and node_id == search_node) else 1,
        borderColor="#FF5733" if (search_node is not None and node_id == search_node) else "#343a40",
        shadow=True,
        widthConstraint={"maximum": 150}
    )
    logging.debug(f"Added node: {label} ({node_id}) with color {color}")

def add_edge(net: Network, src: str, dst: str, relationship: str,
             id_to_label: Dict[str, str], search_node: Optional[str] = None) -> None:
    """
    Add an edge to the pyvis network with styling based on relationship type.
    """
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
        font={"size": 10 if is_search_edge else 8, "align": "middle"},
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
    dark_mode: bool = False,
    community_detection: bool = False
) -> Network:
    """
    Build the pyvis network graph from GraphData with improved interactivity and visuals.
    """
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#f0f2f6" if not dark_mode else "#2c2f33",
        font_color="#343a40" if not dark_mode else "#ffffff"
    )
    net.force_atlas_2based(
        gravity=st.session_state.physics_params.get("gravity", -50),
        central_gravity=st.session_state.physics_params.get("centralGravity", 0.01),
        spring_length=st.session_state.physics_params.get("springLength", 150),
        spring_strength=st.session_state.physics_params.get("springStrength", 0.08)
    )

    added_nodes: Set[str] = set()
    edge_set: Set[Tuple[str, str, str]] = set()
    # Add nodes
    for node in graph_data.nodes:
        if filtered_nodes is not None and node.id not in filtered_nodes:
            logging.debug(f"Skipping node {node.id} due to filtering")
            continue
        color = next((NODE_TYPE_COLORS.get(t, DEFAULT_NODE_COLOR) for t in node.types), DEFAULT_NODE_COLOR)
        if node.id not in added_nodes:
            add_node(net, node.id, id_to_label.get(node.id, node.id), node.types,
                     color, node.metadata, search_node=search_node, show_labels=show_labels)
            added_nodes.add(node.id)
    # Add edges
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

    # Adjust fonts dynamically based on node count
    node_count = len(net.nodes)
    node_font_size = 12 if node_count <= 50 else 10
    edge_font_size = 10 if node_count <= 50 else 8

    default_options = {
        "nodes": {
            "font": {
                "size": node_font_size,
                "face": "Arial",
                "color": "#343a40" if not dark_mode else "#ffffff",
                "strokeWidth": 0
            }
        },
        "edges": {
            "font": {
                "size": edge_font_size,
                "face": "Arial",
                "align": "middle",
                "color": "#343a40" if not dark_mode else "#ffffff"
            },
            "smooth": {
                "type": "continuous"
            }
        },
        "physics": {
            "enabled": False,
            "hierarchicalRepulsion": {
                "centralGravity": 0,
                "springLength": 230,
                "nodeDistance": 210,
                "avoidOverlap": 1
            },
            "minVelocity": 0.75,
            "solver": "hierarchicalRepulsion"
        },
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "zoomView": True,
            "dragNodes": True,
            "multiselect": True,
            "selectConnectedEdges": True
        }
    }
    net.options = default_options

    # Inject custom JS for right-click context menu and drag events.
    custom_js = """
    <script type="text/javascript">
      setTimeout(function() {
          var container = document.getElementById('mynetwork');
          container.addEventListener('contextmenu', function(e) {
              e.preventDefault();
              var pointer = network.getPointer(e);
              var nodeId = network.getNodeAt(pointer);
              if (nodeId) {
                  window.parent.postMessage({type: 'OPEN_MODAL', node: nodeId}, "*");
              }
          });
          network.on("dragEnd", function(params) {
              var positions = {};
              network.body.data.nodes.forEach(function(node) {
                  positions[node.id] = {x: node.x, y: node.y};
              });
              window.parent.postMessage({type: 'UPDATE_POSITIONS', positions: positions}, "*");
          });
      }, 1000);
    </script>
    """
    net.html = net.generate_html() + custom_js
    return net

def create_legends(relationship_colors: Dict[str, str], node_type_colors: Dict[str, str]) -> str:
    """
    Create an HTML legend for relationships and node types.
    """
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
    """Display metadata for a selected node."""
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

def convert_graph_to_jsonld(net: Network) -> Dict[str, Any]:
    """
    Convert the pyvis network graph into JSON‑LD format.
    """
    nodes_dict = {}
    for node in net.nodes:
        node_id = node.get("id")
        nodes_dict[node_id] = {
            "@id": node_id,
            "label": node.get("label", ""),
            "x": node.get("x"),
            "y": node.get("y")
        }
        if "types" in node:
            nodes_dict[node_id]["type"] = node["types"]
    for edge in net.edges:
        source = edge.get("from")
        target = edge.get("to")
        rel = edge.get("label", "").strip()
        if not rel:
            continue
        prop = "ex:" + rel.replace(" ", "")
        triple = {"@id": target}
        if prop in nodes_dict[source]:
            if isinstance(nodes_dict[source][prop], list):
                nodes_dict[source][prop].append(triple)
            else:
                nodes_dict[source][prop] = [nodes_dict[source][prop], triple]
        else:
            nodes_dict[source][prop] = triple

    jsonld = {
        "@context": {
            "label": "http://www.w3.org/2000/01/rdf-schema#label",
            "x": "http://example.org/x",
            "y": "http://example.org/y",
            "type": "@type",
            "ex": "http://example.org/"
        },
        "@graph": list(nodes_dict.values())
    }
    return jsonld

# -----------------------------------------------------------------------------
# Main Streamlit App (UI Module)
# -----------------------------------------------------------------------------
def main() -> None:
    """Main entry point for the Streamlit app."""
    # Allow user to select a theme from sidebar
    theme_choice = st.sidebar.selectbox("Select Theme", options=["Light", "Dark", "Blue"], index=0)
    dark_mode = theme_choice in ["Dark", "Blue"]

    # Custom CSS with enhanced styling
    custom_css = f"""
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
            padding: 0.6rem 1.2rem;
            border-radius: 0.3rem;
            font-size: 1rem;
            transition: background-color 0.3s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background-color: #4cae4c;
        }}
        input, .css-1d391kg {{
            border-radius: 4px;
        }}
        @media (max-width: 768px) {{
            .stApp {{
                padding: 0.5rem;
            }}
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

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
    if "modal_action" not in st.session_state:
        st.session_state.modal_action = None
    if "modal_node" not in st.session_state:
        st.session_state.modal_node = None

    # ---------------- Sidebar: File Upload and Visualization Settings ----------------
    with st.sidebar.expander("File Upload"):
        uploaded_files = st.sidebar.file_uploader(
            label="Upload JSON Files",
            type=["json", "jsonld"],
            accept_multiple_files=True,
            help="Select JSON files describing entities and relationships"
        )
        if uploaded_files:
            file_contents = [uploaded_file.read().decode("utf-8") for uploaded_file in uploaded_files]
            graph_data, id_to_label, errors = parse_entities_from_contents(file_contents)
            st.session_state.graph_data = graph_data
            st.session_state.id_to_label = id_to_label
            try:
                st.session_state.rdf_graph = convert_graph_data_to_rdf(graph_data)
            except Exception as e:
                st.error(f"Error converting graph data to RDF: {e}")
            if errors:
                for error in errors:
                    st.error(error)
    
    with st.sidebar.expander("Visualization Settings"):
        community_detection = st.checkbox("Enable Community Detection", value=False, key="community_detection")
        
        st.subheader("Physics Settings")
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
    
    with st.sidebar.expander("Graph Editing"):
        edit_option = st.radio("Select Action", ("Add Node", "Delete Node", "Modify Node", "Add Edge", "Delete Edge"))
        nodes_list: List[Node] = st.session_state.graph_data.nodes
        if edit_option == "Add Node":
            with st.form("add_node_form"):
                new_node_label = st.text_input("Node Label")
                new_node_type = st.selectbox("Node Type", list(NODE_TYPE_COLORS.keys()))
                submitted = st.form_submit_button("Add Node")
                if submitted:
                    if new_node_label:
                        new_node_id = f"node_{int(time.time())}"  # auto-generated ID
                        new_node = Node(
                            id=new_node_id,
                            label=new_node_label,
                            types=[new_node_type],
                            metadata={"id": new_node_id, "prefLabel": {"en": new_node_label}, "type": [new_node_type]},
                            edges=[]
                        )
                        st.session_state.graph_data.nodes.append(new_node)
                        st.session_state.id_to_label[new_node_id] = new_node_label
                        st.toast(f"Node '{new_node_label}' added!", icon="✅")
                    else:
                        st.error("Please provide a Node Label.")
        elif edit_option == "Delete Node":
            node_ids = [node.id for node in nodes_list]
            if node_ids:
                selected_to_delete = st.selectbox("Select Node to Delete", node_ids)
                if st.button("Delete Node"):
                    st.session_state.graph_data.nodes = [node for node in nodes_list if node.id != selected_to_delete]
                    for node in st.session_state.graph_data.nodes:
                        node.edges = [edge for edge in node.edges if edge.target != selected_to_delete]
                    st.session_state.id_to_label.pop(selected_to_delete, None)
                    st.toast(f"Node '{selected_to_delete}' deleted.", icon="✅")
            else:
                st.info("No nodes available to delete.")
        elif edit_option == "Modify Node":
            node_ids = [node.id for node in nodes_list]
            if node_ids:
                selected_to_modify = st.selectbox("Select Node to Modify", node_ids)
                node_obj = next((node for node in nodes_list if node.id == selected_to_modify), None)
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
                            st.session_state.id_to_label[selected_to_modify] = new_label
                            st.toast(f"Node '{selected_to_modify}' modified.", icon="✅")
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
                        st.toast(f"Edge '{relationship}' from '{source_node}' to '{target_node}' added.", icon="✅")
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
                    st.toast("Edge deleted.", icon="✅")
            else:
                st.info("No edges available to delete.")
    
    with st.sidebar.expander("Manual Node Positioning"):
        if st.session_state.graph_data.nodes:
            unique_nodes = {node.id: node.label for node in st.session_state.graph_data.nodes}
            node_ids = list(unique_nodes.keys())
            selected_node = st.selectbox(
                "Select a Node to Position",
                options=node_ids,
                format_func=lambda x: unique_nodes.get(x, x),
                key="selected_node_control"
            )
            st.session_state.selected_node = selected_node
            current_pos = st.session_state.node_positions.get(selected_node, {"x": 0.0, "y": 0.0})
            with st.form("position_form"):
                x_pos = st.number_input("X Position", value=current_pos["x"], step=10.0)
                y_pos = st.number_input("Y Position", value=current_pos["y"], step=10.0)
                if st.form_submit_button("Set Position"):
                    st.session_state.node_positions[selected_node] = {"x": x_pos, "y": y_pos}
                    st.toast(f"Position for '{unique_nodes[selected_node]}' set to (X: {x_pos}, Y: {y_pos})", icon="✅")
    
    # ---------------- Filter by Entity Types and SPARQL Query ----------------
    if st.session_state.graph_data.nodes:
        all_types = {t for node in st.session_state.graph_data.nodes for t in node.types}
        default_types = [t for t in ["Person", "Organization"] if t in all_types]
        if "filtered_types_control" not in st.session_state:
            st.session_state.filtered_types_control = default_types
        st.session_state.filtered_types = st.sidebar.multiselect(
            "Filter by Entity Types",
            options=list(all_types),
            key="filtered_types_control"
        )
    
    st.session_state.search_term = st.sidebar.text_input(
        "Search for a Node",
        help="Enter the entity name to highlight",
        key="search_term_control",
        value=st.session_state.search_term
    )
    sparql_query_input = st.sidebar.text_area(
        "SPARQL Query",
        help="Enter a SPARQL SELECT query to filter nodes.",
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

    if st.session_state.filtered_types:
        filtered_by_type = {node.id for node in st.session_state.graph_data.nodes
                            if any(t in st.session_state.filtered_types for t in node.types)}
        filtered_nodes = filtered_nodes.intersection(filtered_by_type) if filtered_nodes is not None else filtered_by_type
    
    st.sidebar.markdown(create_legends(RELATIONSHIP_CONFIG, NODE_TYPE_COLORS), unsafe_allow_html=True)
    
    # -------------------- Main Content: Tabbed Interface --------------------
    tabs = st.tabs(["Graph View", "Data View", "SPARQL Query", "About"])
    
    with tabs[0]:
        st.header("🕸️ Network Graph")
        if st.session_state.graph_data.nodes:
            with st.spinner("Generating Network Graph..."):
                net = build_graph(
                    graph_data=st.session_state.graph_data,
                    id_to_label=st.session_state.id_to_label,
                    selected_relationships=st.session_state.selected_relationships,
                    search_node=next(
                        (node.id for node in st.session_state.graph_data.nodes 
                         if node.label.lower() == st.session_state.search_term.lower()),
                        None
                    ),
                    node_positions=st.session_state.node_positions,
                    show_labels=st.session_state.show_labels,
                    filtered_nodes=filtered_nodes,
                    dark_mode=dark_mode,
                    community_detection=community_detection
                )
            # Apply manual positions if available
            if st.session_state.enable_physics:
                for node in net.nodes:
                    pos = st.session_state.node_positions.get(node.get("id"))
                    if pos:
                        node['x'] = pos['x']
                        node['y'] = pos['y']
                        node['fixed'] = True
            else:
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
                except Exception as e:
                    st.error(f"Community detection failed: {e}")
            if isinstance(net.options, dict):
                net.options["configure"] = {
                    "enabled": True,
                    "filter": ["physics"],
                    "showButton": True
                }
            if len(net.nodes) > 50 and st.session_state.show_labels:
                st.info("Graph has many nodes. Consider toggling 'Show Node Labels' off for better readability.")
            try:
                output_path = "network_graph.html"
                net.save_graph(output_path)
                with open(output_path, "r", encoding="utf-8") as f:
                    graph_html = f.read()
                os.remove(output_path)
                st.session_state.graph_html = graph_html
                components.html(graph_html, height=750, scrolling=True)
            except Exception as e:
                st.error(f"Graph generation failed: {e}")
            st.markdown(f"**Total Nodes:** {len(net.nodes)} | **Total Edges:** {len(net.edges)}")
            if st.session_state.selected_node:
                display_node_metadata(st.session_state.selected_node, st.session_state.graph_data, st.session_state.id_to_label)
            else:
                st.markdown("#### Node Metadata")
                st.info("Select a node from the sidebar to view its metadata.")
            # Map view for nodes with geographic coordinates
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
                st.info("No entities with valid coordinates found for map view.")
            # IIIF Viewer integration
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
                    manifest_url = selected_node_obj.metadata.get("image") or selected_node_obj.metadata.get("manifest")
                    if manifest_url and isinstance(manifest_url, (str, list)):
                        if isinstance(manifest_url, list):
                            manifest_url = manifest_url[0]
                        prefix = "https://divinity.contentdm.oclc.org/digital/custom/mirador3?manifest="
                        if manifest_url.startswith(prefix):
                            manifest_url = manifest_url[len(prefix):]
                        if manifest_url.strip() and is_valid_iiif_manifest(manifest_url):
                            st.write(f"Using manifest URL: {manifest_url}")
                            html_code = f'''
<html>
  <head>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/mirador/dist/css/mirador.min.css">
    <script src="https://unpkg.com/mirador/dist/mirador.min.js"></script>
  </head>
  <body>
    <div id="mirador-viewer" style="height: 600px;"></div>
    <script>
      var manifestUrl = {json.dumps(manifest_url)};
      console.log("Manifest URL:", manifestUrl);
      Mirador.viewer({{
        id: 'mirador-viewer',
        windows: [{{ loadedManifest: manifestUrl }}]
      }});
    </script>
  </body>
</html>
'''
                            components.html(html_code, height=650)
                        else:
                            st.info("No valid IIIF manifest found for the selected entity.")
                    else:
                        st.info("No valid IIIF manifest found for the selected entity.")
            else:
                st.info("No entity with a manifest found.")
            # Export Options
            with st.expander("Export Options", expanded=True):
                if "graph_html" in st.session_state:
                    st.download_button(
                        "Download Graph as HTML",
                        data=st.session_state.graph_html,
                        file_name="network_graph.html",
                        mime="text/html"
                    )
                jsonld_data = convert_graph_to_jsonld(net)
                jsonld_str = json.dumps(jsonld_data, indent=2)
                st.download_button(
                    "Download Graph Data as JSON‑LD",
                    data=jsonld_str,
                    file_name="graph_data.jsonld",
                    mime="application/ld+json"
                )
        else:
            st.info("No valid data found. Please check your JSON files.")
    
    with tabs[1]:
        st.header("Data View")
        st.subheader("Graph Nodes")
        if st.session_state.graph_data.nodes:
            data = [{"ID": node.id, "Label": node.label, "Types": ", ".join(node.types)} for node in st.session_state.graph_data.nodes]
            st.dataframe(pd.DataFrame(data))
        else:
            st.info("No data available. Please upload JSON files.")
    
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
            - **Theme Options:** Choose between Light, Dark, and Blue themes.
            - **SPARQL Query Support:** Run queries on your RDF-converted graph and preview results.
            - **IIIF Viewer:** For entities with a manifest, view the associated manifest directly in the app.
            - **Export Options:** Download the graph as HTML or JSON‑LD.
            
            **Usage:**
            - Upload JSON files via the sidebar.
            - Configure graph settings and physics parameters.
            - Edit the graph by adding, deleting, or modifying nodes and edges.
            - Explore your data using the various tabs.
            
            Enjoy exploring your linked data!
            """
        )
    
if __name__ == "__main__":
    main()
