#!/usr/bin/env python
"""
Linked Data Explorer - Enhanced with Modular Refactoring, Automated Testing, CI/CD Integration, and Performance Profiling
Author: Huw Sandaver w/ enhancements and suggestions by ChatGPT
Version: 1.3.5+Refactored
Date: 2025-02-16
"""

# ------------------------------
# Imports and Performance Profiling Decorator
# ------------------------------
import os
import json
import logging
import traceback
import time
import functools
import re  # For sanitizing labels/IDs
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
import plotly.express as px

# Optional streamlit-ace import for SPARQL syntax highlighting
try:
    from streamlit_ace import st_ace
    ace_installed = True
except ImportError:
    ace_installed = False

# Performance profiling decorator to log execution time for key functions
def profile_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logging.info(f"[PROFILE] Function '{func.__name__}' executed in {elapsed:.3f} seconds")
        return result
    return wrapper

# ------------------------------
# Configuration and Constants Module
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="Linked Data Explorer", page_icon="🕸️", layout="wide")

EX = Namespace("http://example.org/")

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

# ------------------------------
# Utility Functions
# ------------------------------
def log_error(message: str) -> None:
    logging.error(message)

def remove_fragment(uri: str) -> str:
    try:
        parsed = urlparse(uri)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ''))
    except Exception as e:
        log_error(f"Error removing fragment from {uri}: {e}")
        return uri

def normalize_relationship_value(rel: str, value: Any) -> Optional[str]:
    if isinstance(value, dict):
        if rel in {
            "spouse", "studentOf", "employedBy", "educatedAt",
            "contributor", "draftsman", "creator", "owner"
        }:
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
    # Ensure required fields exist
    if 'id' not in data or not data['id']:
        raise ValueError("Entity is missing an 'id'.")
    data['id'] = remove_fragment(data.get('id', ''))
    # Set default prefLabel if missing
    if 'prefLabel' not in data or not isinstance(data['prefLabel'], dict) or 'en' not in data['prefLabel']:
        data['prefLabel'] = {'en': data.get('id', 'unknown')}
    # Ensure 'type' is always a list
    if 'type' in data:
        data['type'] = data['type'] if isinstance(data['type'], list) else [data['type']]
    else:
        data['type'] = ["Unknown"]

    for rel in list(data.keys()):
        if rel not in RELATIONSHIP_CONFIG:
            continue
        if rel in ["educatedAt", "employedBy", "dateOfBirth", "dateOfDeath"]:
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
    if not url.startswith("http"):
        return False
    lower_url = url.lower()
    return "iiif" in lower_url and ("manifest" in lower_url or lower_url.endswith("manifest.json"))

def validate_entity(entity: Dict[str, Any]) -> List[str]:
    errors = []
    if 'id' not in entity or not entity['id']:
        errors.append("Missing 'id'.")
    if 'prefLabel' not in entity or not entity['prefLabel'].get('en'):
        errors.append("Missing 'prefLabel' with English label.")
    return errors

def format_metadata(metadata: Dict[str, Any]) -> str:
    """Formats node metadata as a markdown bullet list with clickable links."""
    formatted = ""
    for key, value in metadata.items():
        if key == 'prefLabel':
            continue
        formatted += f"- **{key}**: "
        if isinstance(value, list):
            formatted += "\n" + "\n".join([f"  - {v}" for v in value])
        elif isinstance(value, str):
            if value.startswith("http"):
                formatted += f"[{value}]({value})"
            else:
                formatted += f"{value}"
        elif isinstance(value, dict):
            formatted += "\n" + "\n".join([f"  - **{subkey}**: {subvalue}" for subkey, subvalue in value.items()])
        else:
            formatted += str(value)
        formatted += "\n"
    return formatted

# ------------------------------
# Data Models
# ------------------------------
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
    metadata: Any
    edges: List[Edge] = field(default_factory=list)

@dataclass
class GraphData:
    nodes: List[Node]

# ------------------------------
# Graph Processing
# ------------------------------
@st.cache_data(show_spinner=False)
@profile_time
def parse_entities_from_contents(file_contents: List[str]) -> Tuple[GraphData, Dict[str, str], List[str]]:
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

            # Validate required schema
            validation_errors = validate_entity(normalized)
            if validation_errors:
                errors.append(f"Entity '{subject_id}' errors: " + "; ".join(validation_errors))

            edges: List[Edge] = []
            for rel in RELATIONSHIP_CONFIG:
                values = normalized.get(rel, [])
                if not isinstance(values, list):
                    values = [values]
                for value in values:
                    normalized_id = normalize_relationship_value(rel, value)
                    if normalized_id:
                        edges.append(Edge(source=subject_id, target=normalized_id, relationship=rel))

            new_node = Node(
                id=subject_id, 
                label=label, 
                types=entity_types, 
                metadata=normalized, 
                edges=edges
            )
            nodes.append(new_node)
            id_to_label[subject_id] = label

        except Exception as e:
            err = f"File {idx}: {str(e)}\n{traceback.format_exc()}"
            errors.append(err)
            log_error(err)

    return GraphData(nodes=nodes), id_to_label, errors

@profile_time
def convert_graph_data_to_rdf(graph_data: GraphData) -> RDFGraph:
    g = RDFGraph()
    g.bind("ex", EX)

    for node in graph_data.nodes:
        subject = URIRef(node.id)
        label = node.metadata.get("prefLabel", {}).get("en", node.id)
        g.add((subject, RDFS.label, Literal(label, lang="en")))

        # Insert RDF types
        for t in node.types:
            g.add((subject, RDF.type, EX[t]))

        # Insert other metadata
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

        # Insert edges
        for edge in node.edges:
            g.add((subject, EX[edge.relationship], URIRef(edge.target)))

    return g

def run_sparql_query(query: str, rdf_graph: RDFGraph) -> Set[str]:
    result = rdf_graph.query(query, initNs={'rdf': RDF, 'ex': EX})
    return {str(row[0]) for row in result if row[0] is not None}

@st.cache_data(show_spinner=False)
@profile_time
def compute_centrality_measures(graph_data: GraphData) -> Dict[str, Dict[str, float]]:
    """
    Computes degree and betweenness centrality measures for nodes in the graph.
    """
    G = nx.DiGraph()

    for node in graph_data.nodes:
        G.add_node(node.id)

    for node in graph_data.nodes:
        for edge in node.edges:
            G.add_edge(edge.source, edge.target)

    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)

    centrality = {}
    for node in G.nodes():
        centrality[node] = {
            "degree": degree.get(node, 0),
            "betweenness": betweenness.get(node, 0)
        }
    return centrality

def get_edge_relationship(source: str, target: str, graph_data: GraphData) -> List[str]:
    relationships = []
    for node in graph_data.nodes:
        if node.id == source:
            for edge in node.edges:
                if edge.target == target:
                    relationships.append(edge.relationship)
    return relationships

# ------------------------------
# Remote SPARQL Endpoint
# ------------------------------
def load_data_from_sparql(endpoint_url: str) -> Tuple[GraphData, Dict[str, str], List[str]]:
    errors = []
    nodes_dict = {}
    id_to_label = {}

    try:
        from rdflib.plugins.stores.sparqlstore import SPARQLStore
        store = SPARQLStore(endpoint_url)
        rdf_graph = RDFGraph(store=store)
        query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100"
        results = rdf_graph.query(query)

        for row in results:
            s = str(row.s)
            p = str(row.p)
            o = str(row.o)

            if s not in nodes_dict:
                nodes_dict[s] = {"id": s, "prefLabel": {"en": s}, "type": ["Unknown"], "metadata": {}}

            if p in nodes_dict[s]["metadata"]:
                if isinstance(nodes_dict[s]["metadata"][p], list):
                    nodes_dict[s]["metadata"][p].append(o)
                else:
                    nodes_dict[s]["metadata"][p] = [nodes_dict[s]["metadata"][p], o]
            else:
                nodes_dict[s]["metadata"][p] = o

            if p.endswith("label"):
                nodes_dict[s]["prefLabel"] = {"en": o}

        nodes = []
        for s, data in nodes_dict.items():
            node = Node(
                id=data["id"],
                label=data["prefLabel"]["en"],
                types=data.get("type", ["Unknown"]),
                metadata=data["metadata"],
                edges=[]
            )
            nodes.append(node)
            id_to_label[s] = data["prefLabel"]["en"]

        graph_data = GraphData(nodes=nodes)

    except Exception as e:
        errors.append(f"Error loading from SPARQL endpoint: {e}")
        graph_data = GraphData(nodes=[])

    return graph_data, id_to_label, errors

# ------------------------------
# Graph Building
# ------------------------------
def add_node(
    net: Network,
    node_id: str,
    label: str,
    entity_types: List[str],
    color: str,
    metadata: Dict[str, Any],
    search_nodes: Optional[List[str]] = None,
    show_labels: bool = True,
    custom_size: Optional[int] = None
) -> None:
    node_title = f"{label}\nTypes: {', '.join(entity_types)}"
    description = ""

    if "description" in metadata:
        if isinstance(metadata["description"], dict):
            description = metadata["description"].get("en", "")
        elif isinstance(metadata["description"], str):
            description = metadata["description"]

    if description:
        node_title += f"\nDescription: {description}"

    size = custom_size if custom_size is not None else (20 if (search_nodes and node_id in search_nodes) else 15)

    net.add_node(
        node_id,
        label=label if show_labels else "",
        title=node_title,
        color=color,
        shape=NODE_TYPE_SHAPES.get(entity_types[0], "dot") if entity_types else "dot",
        size=size,
        font={"size": 12 if size >= 20 else 10, "face": "Arial", "color": "#343a40"},
        borderWidth=2 if (search_nodes and node_id in search_nodes) else 1,
        borderColor="#FF5733" if (search_nodes and node_id in search_nodes) else "#343a40",
        shadow=True,
        widthConstraint={"maximum": 150}
    )
    logging.debug(f"Added node: {label} ({node_id}) with color {color}")

def add_edge(
    net: Network,
    src: str,
    dst: str,
    relationship: str,
    id_to_label: Dict[str, str],
    search_nodes: Optional[List[str]] = None,
    custom_width: Optional[int] = None,
    custom_color: Optional[str] = None
) -> None:
    is_search_edge = search_nodes is not None and (src in search_nodes or dst in search_nodes)
    edge_color = custom_color if custom_color is not None else RELATIONSHIP_CONFIG.get(relationship, "#A9A9A9")
    label_text = " ".join(word.capitalize() for word in relationship.split('_'))
    width = custom_width if custom_width is not None else (3 if is_search_edge else 2)

    net.add_edge(
        src,
        dst,
        label=label_text,
        color=edge_color,
        width=width,
        arrows='to',
        title=f"{label_text}: {id_to_label.get(src, src)} → {id_to_label.get(dst, dst)}",
        font={"size": 10 if is_search_edge else 8, "align": "middle"},
        smooth={'enabled': True, 'type': 'continuous'}
    )
    logging.debug(f"Added edge: {src} --{label_text}--> {dst}")

@profile_time
def build_graph(
    graph_data: GraphData,
    id_to_label: Dict[str, str],
    selected_relationships: List[str],
    search_nodes: Optional[List[str]] = None,
    node_positions: Optional[Dict[str, Dict[str, float]]] = None,
    show_labels: bool = True,
    filtered_nodes: Optional[Set[str]] = None,
    community_detection: bool = False,
    centrality: Optional[Dict[str, Dict[str, float]]] = None,
    path_nodes: Optional[List[str]] = None
) -> Network:
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#f0f2f6",
        font_color="#343a40"
    )

    net.force_atlas_2based(
        gravity=st.session_state.physics_params.get("gravity", -50),
        central_gravity=st.session_state.physics_params.get("centralGravity", 0.01),
        spring_length=st.session_state.physics_params.get("springLength", 150),
        spring_strength=st.session_state.physics_params.get("springStrength", 0.08)
    )

    added_nodes: Set[str] = set()
    edge_set: Set[Tuple[str, str, str]] = set()
    path_edge_set = set(zip(path_nodes, path_nodes[1:])) if path_nodes else set()

    for node in graph_data.nodes:
        if filtered_nodes is not None and node.id not in filtered_nodes:
            logging.debug(f"Skipping node {node.id} due to filtering")
            continue

        color = next(
            (NODE_TYPE_COLORS.get(t, DEFAULT_NODE_COLOR) for t in node.types),
            DEFAULT_NODE_COLOR
        )

        custom_size = None
        if centrality and node.id in centrality:
            custom_size = int(15 + centrality[node.id]["degree"] * 30)

        if path_nodes and node.id in path_nodes:
            custom_size = max(custom_size or 15, 25)

        if node.id not in added_nodes:
            add_node(
                net,
                node.id,
                id_to_label.get(node.id, node.id),
                node.types,
                color,
                node.metadata,
                search_nodes=search_nodes,
                show_labels=show_labels,
                custom_size=custom_size
            )
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
                add_node(
                    net,
                    edge.target,
                    target_label,
                    ["Unknown"],
                    DEFAULT_NODE_COLOR,
                    {},
                    search_nodes=search_nodes,
                    show_labels=show_labels
                )
                added_nodes.add(edge.target)

            if (edge.source, edge.target, edge.relationship) not in edge_set:
                if path_nodes and (edge.source, edge.target) in path_edge_set:
                    custom_width = 4
                    custom_color = "#FF0000"
                else:
                    custom_width = None
                    custom_color = None

                add_edge(
                    net,
                    edge.source,
                    edge.target,
                    edge.relationship,
                    id_to_label,
                    search_nodes=search_nodes,
                    custom_width=custom_width,
                    custom_color=custom_color
                )
                edge_set.add((edge.source, edge.target, edge.relationship))

    node_count = len(net.nodes)
    node_font_size = 12 if node_count <= 50 else 10
    edge_font_size = 10 if node_count <= 50 else 8

    default_options = {
        "nodes": {
            "font": {
                "size": node_font_size,
                "face": "Arial",
                "color": "#343a40",
                "strokeWidth": 0
            }
        },
        "edges": {
            "font": {
                "size": edge_font_size,
                "face": "Arial",
                "align": "middle",
                "color": "#343a40"
            },
            "smooth": {"type": "continuous"}
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

    if node_positions:
        for node in net.nodes:
            pos = node_positions.get(node.get("id"))
            if pos:
                node['x'] = pos['x']
                node['y'] = pos['y']
                node['fixed'] = True
                node['physics'] = False

    if community_detection:
        G = nx.Graph()
        for node in net.nodes:
            G.add_node(node["id"])
        for edge in net.edges:
            G.add_edge(edge["from"], edge["to"])
        try:
            communities = custom_community_detection(G, max_iter=20)
            if communities:
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
            else:
                st.info("No communities detected.")
        except Exception as e:
            st.error(f"Community detection failed: {e}")

    net.html = net.generate_html() + custom_js
    return net

def custom_community_detection(G: nx.Graph, max_iter: int = 5) -> List[Set[str]]:
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    best_modularity = nx.algorithms.community.modularity(G, communities)

    for _ in range(max_iter):
        weak_nodes = set()
        new_communities = []

        for comm in communities:
            comm = set(comm)
            if len(comm) >= 3:
                triangles = nx.triangles(G.subgraph(comm))
            else:
                triangles = {v: 0 for v in comm}

            strong_nodes = set()
            for v in comm:
                total_deg = G.degree(v)
                internal_deg = sum(1 for nb in G.neighbors(v) if nb in comm)
                if total_deg == 0:
                    strong_nodes.add(v)
                    continue
                if internal_deg <= total_deg / 2:
                    weak_nodes.add(v)
                elif len(comm) >= 3 and triangles.get(v, 0) == 0:
                    weak_nodes.add(v)
                else:
                    strong_nodes.add(v)

            if strong_nodes:
                new_communities.append(strong_nodes)

        if not weak_nodes:
            break

        G_mod = G.copy()
        G_mod.remove_nodes_from(weak_nodes)

        if len(G_mod.nodes()) > 0:
            mod_communities = list(nx.algorithms.community.greedy_modularity_communities(G_mod))
        else:
            mod_communities = []

        for w in weak_nodes:
            mod_communities.append({w})

        new_modularity = nx.algorithms.community.modularity(G, mod_communities)
        if new_modularity > best_modularity:
            best_modularity = new_modularity
            communities = mod_communities
        else:
            break

    return communities

# ------------------------------
# Conversion Functions
# ------------------------------
def convert_graph_to_jsonld(net: Network) -> Dict[str, Any]:
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

    return {
        "@context": {
            "label": "http://www.w3.org/2000/01/rdf-schema#label",
            "x": "http://example.org/x",
            "y": "http://example.org/y",
            "type": "@type",
            "ex": "http://example.org/"
        },
        "@graph": list(nodes_dict.values())
    }

# ------------------------------
# Streamlit UI (Main)
# ------------------------------
def main() -> None:
    custom_css = """
    <style>
        .stApp {
            max-width: 1600px;
            padding: 1rem;
            background-color: #fafafa;
            color: #343a40;
        }
        section[data-testid="stSidebar"] > div {
            background-color: #f8f9fa;
            padding: 1rem;
        }
        h1, h2, h3, h4, h5 {
            color: #333;
        }
        .stButton > button, .stDownloadButton > button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 0.3rem;
            font-size: 1rem;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background-color: #0056b3;
        }
        .css-1d391kg, .stTextInput, .stSelectbox, .stTextArea {
            border-radius: 4px;
        }
        .stTextInput > label, .stSelectbox > label, .stTextArea > label {
            font-size: 0.9rem;
            font-weight: 600;
        }
        .stExpander > label {
            font-size: 0.95rem;
            font-weight: 700;
        }
        .stTabs [role="tab"] {
            font-weight: 600;
        }
        header[data-testid="stHeader"] {
            background: #f8f9fa;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    st.title("Linked Data Explorer")
    st.caption("Visualize and navigate your linked data.")

    with st.expander("How to Use This App"):
        st.write("1. Upload your JSON or JSON‑LD files in the **File Upload** section on the sidebar.")
        st.write("2. Adjust visualization settings like physics, filtering, and (optionally) enable centrality measures.")
        st.write("3. (Optional) Run SPARQL queries to narrow down specific nodes or load data from a remote SPARQL endpoint.")
        st.write("4. Explore the graph in the **Graph View** tab below!")
        st.write("5. Manually set node positions using the sidebar.")

    if not ace_installed:
        st.sidebar.info("streamlit-ace not installed; SPARQL syntax highlighting will be disabled.")

    # Initialize session state
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
    if "centrality_measures" not in st.session_state:
        st.session_state.centrality_measures = None
    if "shortest_path" not in st.session_state:
        st.session_state.shortest_path = None
    if "property_filter" not in st.session_state:
        st.session_state.property_filter = {"property": "", "value": ""}

    # ---------------- Sidebar
    with st.sidebar.expander("File Upload"):
        uploaded_files = st.sidebar.file_uploader(
            label="Upload JSON Files",
            type=["json", "jsonld"],
            accept_multiple_files=True,
            help="Select JSON files describing entities and relationships"
        )
        if uploaded_files:
            file_contents = [f.read().decode("utf-8") for f in uploaded_files]
            graph_data, id_to_label, errors = parse_entities_from_contents(file_contents)
            st.session_state.graph_data = graph_data
            st.session_state.id_to_label = id_to_label
            try:
                st.session_state.rdf_graph = convert_graph_data_to_rdf(graph_data)
            except Exception as e:
                st.error(f"Error converting graph data to RDF: {e}")
            if errors:
                with st.expander("Data Validation"):
                    for error in errors:
                        st.error(error)
        
        st.markdown("---")
        st.subheader("Remote SPARQL Endpoint")
        endpoint_url = st.text_input("Enter SPARQL Endpoint URL", key="sparql_endpoint_url")
        if st.button("Load from SPARQL Endpoint"):
            remote_graph, remote_id_to_label, sparql_errors = load_data_from_sparql(endpoint_url)
            if sparql_errors:
                for err in sparql_errors:
                    st.error(err)
            else:
                st.success("Data loaded from SPARQL endpoint.")
                st.session_state.graph_data = remote_graph
                st.session_state.id_to_label = remote_id_to_label
                try:
                    st.session_state.rdf_graph = convert_graph_data_to_rdf(remote_graph)
                except Exception as e:
                    st.error(f"Error converting SPARQL data to RDF: {e}")

    with st.sidebar.expander("Visualization Settings"):
        community_detection = st.checkbox("Enable Community Detection", value=False, key="community_detection")
        
        physics_presets = {
            "Default (Balanced)": {"gravity": -50, "centralGravity": 0.01, "springLength": 150, "springStrength": 0.08},
            "High Gravity (Clustering)": {"gravity": -100, "centralGravity": 0.05, "springLength": 100, "springStrength": 0.15},
            "No Physics (Manual Layout)": {"gravity": 0, "centralGravity": 0, "springLength": 150, "springStrength": 0},
            "Custom": st.session_state.physics_params
        }
        preset_name = st.selectbox("Physics Presets", list(physics_presets.keys()), index=0, key="physics_preset")
        if preset_name != "Custom":
            st.session_state.physics_params = physics_presets[preset_name]
            st.info("Physics parameters set to preset: " + preset_name)
        else:
            st.subheader("Advanced Physics Settings")
            st.session_state.physics_params["gravity"] = st.number_input(
                "Gravity",
                value=float(st.session_state.physics_params.get("gravity", -50)),
                step=1.0,
                key="gravity_input"
            )
            st.session_state.physics_params["centralGravity"] = st.number_input(
                "Central Gravity",
                value=st.session_state.physics_params.get("centralGravity", 0.01),
                step=0.01,
                key="centralGravity_input"
            )
            st.session_state.physics_params["springLength"] = st.number_input(
                "Spring Length",
                value=float(st.session_state.physics_params.get("springLength", 150)),
                step=1.0,
                key="springLength_input"
            )
            st.session_state.physics_params["springStrength"] = st.number_input(
                "Spring Strength",
                value=st.session_state.physics_params.get("springStrength", 0.08),
                step=0.01,
                key="springStrength_input"
            )

        enable_centrality = st.checkbox("Display Centrality Measures", value=False, key="centrality_enabled")
        if enable_centrality and st.session_state.graph_data.nodes:
            st.session_state.centrality_measures = compute_centrality_measures(st.session_state.graph_data)
            st.info("Centrality measures computed.")

    with st.sidebar.expander("Graph Editing"):
        ge_tabs = st.tabs(["Add Node", "Delete Node", "Modify Node", "Add Edge", "Delete Edge"])
        nodes_list = st.session_state.graph_data.nodes

        with ge_tabs[0]:
            with st.form("add_node_form"):
                new_label = st.text_input("Node Label")
                new_type = st.selectbox("Node Type", list(NODE_TYPE_COLORS.keys()))
                if st.form_submit_button("Add Node"):
                    if new_label:
                        nid = f"node_{int(time.time())}"
                        new_node = Node(
                            id=nid,
                            label=new_label,
                            types=[new_type],
                            metadata={
                                "id": nid,
                                "prefLabel": {"en": new_label},
                                "type": [new_type]
                            }
                        )
                        st.session_state.graph_data.nodes.append(new_node)
                        st.session_state.id_to_label[nid] = new_label
                        st.success(f"Node '{new_label}' added!")
                    else:
                        st.error("Please provide a Node Label.")

        with ge_tabs[1]:
            nid_list = [n.id for n in nodes_list]
            if nid_list:
                node_to_delete = st.selectbox("Select Node to Delete", nid_list)
                if st.button("Delete Node"):
                    st.session_state.graph_data.nodes = [
                        n for n in nodes_list if n.id != node_to_delete
                    ]
                    for node in st.session_state.graph_data.nodes:
                        node.edges = [e for e in node.edges if e.target != node_to_delete]
                    st.session_state.id_to_label.pop(node_to_delete, None)
                    st.success(f"Node '{node_to_delete}' deleted.")
            else:
                st.info("No nodes to delete.")

        with ge_tabs[2]:
            nid_list = [n.id for n in nodes_list]
            if nid_list:
                node_to_modify = st.selectbox("Select Node to Modify", nid_list)
                node_obj = next((n for n in nodes_list if n.id == node_to_modify), None)
                if node_obj:
                    with st.form("modify_node_form"):
                        new_label = st.text_input("New Label", node_obj.label)
                        current_type = node_obj.types[0] if node_obj.types else "Unknown"
                        new_type = st.selectbox(
                            "New Type",
                            list(NODE_TYPE_COLORS.keys()),
                            index=(list(NODE_TYPE_COLORS.keys()).index(current_type)
                                    if current_type in NODE_TYPE_COLORS else 0)
                        )
                        if st.form_submit_button("Modify Node"):
                            node_obj.label = new_label
                            node_obj.types = [new_type]
                            node_obj.metadata["prefLabel"]["en"] = new_label
                            st.session_state.id_to_label[node_to_modify] = new_label
                            st.success(f"Node '{node_to_modify}' modified.")
            else:
                st.info("No nodes to modify.")

        with ge_tabs[3]:
            if nodes_list:
                with st.form("add_edge_form"):
                    src_node = st.selectbox("Source Node", [n.id for n in nodes_list])
                    tgt_node = st.selectbox("Target Node", [n.id for n in nodes_list])
                    rel = st.selectbox("Relationship", list(RELATIONSHIP_CONFIG.keys()))
                    if st.form_submit_button("Add Edge"):
                        for n in nodes_list:
                            if n.id == src_node:
                                n.edges.append(Edge(source=src_node, target=tgt_node, relationship=rel))
                        st.success(f"Edge '{rel}' from '{src_node}' to '{tgt_node}' added.")
            else:
                st.info("No nodes to add edges to.")

        with ge_tabs[4]:
            all_edges = []
            for n in nodes_list:
                for e in n.edges:
                    all_edges.append((e.source, e.target, e.relationship))
            if all_edges:
                edge_to_delete = st.selectbox("Select Edge to Delete", all_edges)
                if st.button("Delete Edge"):
                    for n in nodes_list:
                        if n.id == edge_to_delete[0]:
                            n.edges = [
                                e for e in n.edges
                                if (e.source, e.target, e.relationship) != edge_to_delete
                            ]
                    st.success("Edge deleted.")
            else:
                st.info("No edges to delete.")

    with st.sidebar.expander("Manual Node Positioning"):
        if st.session_state.graph_data.nodes:
            unique_nodes = {n.id: n.label for n in st.session_state.graph_data.nodes}
            sel_node = st.selectbox(
                "Select a Node to Position",
                list(unique_nodes.keys()),
                format_func=lambda x: unique_nodes[x],
                key="selected_node_control"
            )
            st.session_state.selected_node = sel_node
            cur_pos = st.session_state.node_positions.get(sel_node, {"x": 0.0, "y": 0.0})
            with st.form("position_form"):
                x_val = st.number_input("X Position", value=cur_pos["x"], step=10.0)
                y_val = st.number_input("Y Position", value=cur_pos["y"], step=10.0)
                if st.form_submit_button("Set Position"):
                    st.session_state.node_positions[sel_node] = {"x": x_val, "y": y_val}
                    st.success(f"Position for '{unique_nodes[sel_node]}' set to (X: {x_val}, Y: {y_val})")

    with st.sidebar.expander("Advanced Filtering"):
        st.subheader("Property-based Filtering")
        prop_name = st.text_input("Property Name", key="filter_prop_name")
        prop_value = st.text_input("Property Value", key="filter_prop_value")
        if st.button("Apply Property Filter"):
            st.session_state.property_filter = {"property": prop_name, "value": prop_value}
            st.success("Property filter applied.")

        all_rels = set()
        for n in st.session_state.graph_data.nodes:
            for e in n.edges:
                all_rels.add(e.relationship)
        chosen_rels = st.multiselect("Select Relationship Types", sorted(all_rels), default=list(all_rels))
        st.session_state.selected_relationships = chosen_rels if chosen_rels else list(RELATIONSHIP_CONFIG.keys())

        st.subheader("Filter by Node Type")
        unique_types = sorted({t for n in st.session_state.graph_data.nodes for t in n.types})
        chosen_types = st.multiselect("Select Node Types", options=unique_types, default=unique_types, key="filter_node_types")
        st.session_state.filtered_types = chosen_types

    st.session_state.sparql_query = st.sidebar.text_area(
        "SPARQL Query",
        help="Enter a SPARQL SELECT query to filter nodes.",
        key="sparql_query_control",
        value=st.session_state.sparql_query
    )
    if st.session_state.sparql_query.strip():
        st.sidebar.info("Query Running...")
        try:
            filtered_nodes = run_sparql_query(st.session_state.sparql_query, st.session_state.rdf_graph)
            st.sidebar.success(f"Query Successful: {len(filtered_nodes)} result(s) found.")
            st.sidebar.dataframe(pd.DataFrame(list(filtered_nodes), columns=["Node ID"]))
        except Exception as e:
            st.sidebar.error(f"SPARQL Query failed: {e}")
            filtered_nodes = None
    else:
        filtered_nodes = None

    if st.session_state.filtered_types:
        filter_by_type = {
            n.id for n in st.session_state.graph_data.nodes if any(t in st.session_state.filtered_types for t in n.types)
        }
        filtered_nodes = filtered_nodes.intersection(filter_by_type) if filtered_nodes is not None else filter_by_type

    with st.sidebar.expander("Graph Pathfinding"):
        if st.session_state.graph_data.nodes:
            source_pf = st.selectbox("Source Node", [n.id for n in st.session_state.graph_data.nodes], key="pf_source")
            target_pf = st.selectbox("Target Node", [n.id for n in st.session_state.graph_data.nodes], key="pf_target")
            if st.button("Find Shortest Path"):
                try:
                    G_pf = nx.DiGraph()
                    for n in st.session_state.graph_data.nodes:
                        G_pf.add_node(n.id)
                    for n in st.session_state.graph_data.nodes:
                        for e in n.edges:
                            G_pf.add_edge(e.source, e.target)
                    sp = nx.shortest_path(G_pf, source=source_pf, target=target_pf)
                    st.session_state.shortest_path = sp
                    st.success(f"Shortest path found with {len(sp)-1} edges.")
                except Exception as e:
                    st.session_state.shortest_path = None
                    st.error(f"Pathfinding failed: {e}")
        else:
            st.info("No nodes available for pathfinding.")

    tabs = st.tabs(["Graph View", "Data View", "Centrality Measures", "SPARQL Query", "Timeline", "About"])

    with tabs[0]:
        st.header("Network Graph")
        if st.session_state.graph_data.nodes:
            with st.spinner("Generating Network Graph..."):
                if st.session_state.search_term.strip():
                    search_nodes = [
                        n.id for n in st.session_state.graph_data.nodes
                        if st.session_state.search_term.lower() in n.label.lower()
                    ]
                else:
                    search_nodes = None

                if st.session_state.property_filter["property"] and st.session_state.property_filter["value"]:
                    prop = st.session_state.property_filter["property"]
                    val = st.session_state.property_filter["value"].lower()
                    prop_nodes = {
                        n.id for n in st.session_state.graph_data.nodes
                        if prop in n.metadata and val in str(n.metadata[prop]).lower()
                    }
                    filtered_nodes = filtered_nodes.intersection(prop_nodes) if filtered_nodes is not None else prop_nodes

                net = build_graph(
                    graph_data=st.session_state.graph_data,
                    id_to_label=st.session_state.id_to_label,
                    selected_relationships=st.session_state.selected_relationships,
                    search_nodes=search_nodes,
                    node_positions=st.session_state.node_positions,
                    show_labels=st.session_state.show_labels,
                    filtered_nodes=filtered_nodes,
                    community_detection=community_detection,
                    centrality=st.session_state.centrality_measures,
                    path_nodes=st.session_state.shortest_path
                )

            if community_detection:
                st.info("Community detection applied to node colors.")
            if len(net.nodes) > 50 and st.session_state.show_labels:
                st.info("Graph has many nodes. Consider toggling 'Show Node Labels' off for better readability.")

            try:
                st.session_state.graph_html = net.html
                components.html(net.html, height=750, scrolling=True)
            except Exception as e:
                st.error(f"Graph generation failed: {e}")

            st.markdown(create_legends(RELATIONSHIP_CONFIG, NODE_TYPE_COLORS), unsafe_allow_html=True)
            st.markdown(f"**Total Nodes:** {len(net.nodes)} | **Total Edges:** {len(net.edges)}")

            st.markdown("#### Node Metadata")
            if st.session_state.selected_node:
                node_obj = next((n for n in st.session_state.graph_data.nodes if n.id == st.session_state.selected_node), None)
                if node_obj:
                    md_content = f"**Label:** {st.session_state.id_to_label.get(node_obj.id, node_obj.id)}\n\n"
                    if st.session_state.centrality_measures and node_obj.id in st.session_state.centrality_measures:
                        c_meas = st.session_state.centrality_measures[node_obj.id]
                        md_content += f"- **Degree Centrality:** {c_meas['degree']:.3f}\n"
                        md_content += f"- **Betweenness Centrality:** {c_meas['betweenness']:.3f}\n\n"
                    md_content += format_metadata(node_obj.metadata)
                    st.markdown(md_content)
                else:
                    st.write("No metadata available for this node.")
            else:
                st.info("Select a node from the sidebar to view its metadata.")

            place_locations = []
            for n in st.session_state.graph_data.nodes:
                if isinstance(n.metadata, dict) and "geographicCoordinates" in n.metadata:
                    coords = n.metadata["geographicCoordinates"]
                    if isinstance(coords, list):
                        coords = coords[0]
                    if isinstance(coords, str) and coords.startswith("Point(") and coords.endswith(")"):
                        coords = coords[6:-1].strip()
                        parts = coords.split()
                        if len(parts) == 2:
                            try:
                                lon, lat = map(float, parts)
                                place_locations.append({"lat": lat, "lon": lon, "label": n.label})
                            except ValueError:
                                logging.error(f"Invalid coordinates for node {n.id}: {coords}")
                elif isinstance(n.metadata, dict) and "latitude" in n.metadata and "longitude" in n.metadata:
                    lat = n.metadata.get("latitude")
                    lon = n.metadata.get("longitude")
                    if lat is not None and lon is not None:
                        try:
                            place_locations.append({"lat": float(lat), "lon": float(lon), "label": n.label})
                        except ValueError:
                            logging.error(f"Invalid numeric coordinates for node {n.id}: {lat}, {lon}")

            if place_locations:
                st.subheader("Map View of Entities with Coordinates")
                df_places = pd.DataFrame(place_locations)
                st.map(df_places)
            else:
                st.info("No entities with valid coordinates found for map view.")

            st.subheader("IIIF Viewer")
            iiif_nodes = [
                n for n in st.session_state.graph_data.nodes
                if isinstance(n.metadata, dict) and ("image" in n.metadata or "manifest" in n.metadata)
            ]
            if iiif_nodes:
                sel_iiif = st.selectbox(
                    "Select an entity with a manifest for IIIF Viewer",
                    [n.id for n in iiif_nodes],
                    format_func=lambda x: st.session_state.id_to_label.get(x, x)
                )
                node_iiif = next((n for n in iiif_nodes if n.id == sel_iiif), None)
                if node_iiif:
                    manifest_url = node_iiif.metadata.get("image") or node_iiif.metadata.get("manifest")
                    if manifest_url:
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

            if st.session_state.shortest_path:
                st.subheader("Shortest Path Details")
                path_list = st.session_state.shortest_path
                path_text = " → ".join([st.session_state.id_to_label.get(x, x) for x in path_list])
                path_text = re.sub(r'[^\x20-\x7E]+', '', path_text)
                if len(path_text) > 1000:
                    path_text = path_text[:1000] + "... [truncated]"
                st.text_area("Shortest Path", value=path_text, height=100)
                rel_text = ""
                for i in range(len(path_list) - 1):
                    rels = get_edge_relationship(path_list[i], path_list[i+1], st.session_state.graph_data)
                    rel_text += f"{st.session_state.id_to_label.get(path_list[i], path_list[i])} -- {', '.join(rels)} --> "
                rel_text += st.session_state.id_to_label.get(path_list[-1], path_list[-1])
                rel_text = re.sub(r'[^\x20-\x7E]+', '', rel_text)
                if len(rel_text) > 1000:
                    rel_text = rel_text[:1000] + "... [truncated]"
                st.text_area("Path Relationships", value=rel_text, height=100)

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
            data_rows = []
            for n in st.session_state.graph_data.nodes:
                safe_id = re.sub(r'[^\x20-\x7E]+', '', n.id)
                safe_label = re.sub(r'[^\x20-\x7E]+', '', n.label)
                safe_types = re.sub(r'[^\x20-\x7E]+', '', ", ".join(n.types))
                data_rows.append({
                    "ID": safe_id,
                    "Label": safe_label,
                    "Types": safe_types
                })
            df_nodes = pd.DataFrame(data_rows)
            st.dataframe(df_nodes)
            csv_data = df_nodes.to_csv(index=False).encode('utf-8')
            st.download_button("Download Nodes as CSV", data=csv_data, file_name="nodes.csv", mime="text/csv")
        else:
            st.info("No data available. Please upload JSON files.")
    
    with tabs[2]:
        st.header("Centrality Measures")
        if st.session_state.centrality_measures:
            # Convert centrality measures dict to a pandas DataFrame
            centrality_df = pd.DataFrame.from_dict(st.session_state.centrality_measures, orient='index')
            centrality_df = centrality_df.reset_index().rename(columns={"index": "Node ID"})
            st.dataframe(centrality_df)
            csv_data = centrality_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Centrality Measures as CSV", data=csv_data, file_name="centrality_measures.csv", mime="text/csv")
        else:
            st.info("Centrality measures have not been computed yet. Please enable 'Display Centrality Measures' in the visualization settings.")
    
    with tabs[3]:
        st.header("SPARQL Query")
        st.markdown("Enter a SPARQL SELECT query in the sidebar and view the results here.")
        if st.session_state.sparql_query.strip():
            try:
                query_results = run_sparql_query(st.session_state.sparql_query, st.session_state.rdf_graph)
                st.success(f"Query returned {len(query_results)} result(s).")
                st.dataframe(pd.DataFrame(list(query_results), columns=["Node ID"]))
            except Exception as e:
                st.error(f"SPARQL Query failed: {e}")
        else:
            st.info("No query entered.")

    with tabs[4]:
        st.header("Timeline View")
        # Static timeline implementation
        timeline_data = []
        for n in st.session_state.graph_data.nodes:
            dob = n.metadata.get("dateOfBirth")
            if isinstance(dob, list) and dob:
                dval = dob[0].get("time:inXSDDateTimeStamp", {}).get("@value")
                if dval:
                    timeline_data.append({"Label": n.label, "Event": "Birth", "Date": dval})
            dod = n.metadata.get("dateOfDeath")
            if isinstance(dod, list) and dod:
                dval = dod[0].get("time:inXSDDateTimeStamp", {}).get("@value")
                if dval:
                    timeline_data.append({"Label": n.label, "Event": "Death", "Date": dval})
            for rel in ["educatedAt", "employedBy"]:
                events = n.metadata.get(rel)
                if events:
                    if not isinstance(events, list):
                        events = [events]
                    for ev in events:
                        if isinstance(ev, dict):
                            start = ev.get("startDate")
                            if start:
                                val_start = start.get("time:inXSDDateTimeStamp", {}).get("@value")
                                if val_start:
                                    timeline_data.append({
                                        "Label": n.label,
                                        "Event": f"{rel} Start",
                                        "Date": val_start
                                    })
                            end = ev.get("endDate")
                            if end:
                                val_end = end.get("time:inXSDDateTimeStamp", {}).get("@value")
                                if val_end:
                                    timeline_data.append({
                                        "Label": n.label,
                                        "Event": f"{rel} End",
                                        "Date": val_end
                                    })
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            from dateutil.parser import parse
            df_timeline["Date"] = df_timeline["Date"].apply(lambda x: parse(x))
            fig_static = px.scatter(
                df_timeline,
                x="Date",
                y="Label",
                color="Event",
                hover_data=["Event", "Date"],
                title="Entity Timeline (Scatter Plot)"
            )
            fig_static.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_static, use_container_width=True)
        else:
            st.info("No timeline data available.")

    with tabs[5]:
        st.header("About Linked Data Explorer")
        st.markdown(
            """
            ### Explore Relationships Between Entities
            Upload multiple JSON files representing entities and generate an interactive network.
            Use the sidebar to filter relationships, search for nodes, set manual positions,
            and edit the graph directly.
            
            **Features:**
            - **Tabbed Interface:** Separate views for the graph, raw data, centrality measures, SPARQL queries, timeline, and about information.
            - **Dynamic Graph Visualization:** Interactive network graph with manual node positioning, centrality measures, and optional community detection.
            - **Physics Presets:** Easily switch between default, high gravity, no physics, or custom physics settings.
            - **SPARQL Query Support:** Run queries on your RDF-converted graph (syntax highlighting if streamlit-ace is installed).
            - **IIIF Viewer:** View IIIF manifests for applicable entities.
            - **Advanced Filtering:** Filter nodes by properties, relationship types, and node types.
            - **Pathfinding:** Find and visually highlight the shortest path between nodes.
            - **Export Options:** Download the graph as HTML, JSON‑LD, or CSV.
            
            **Version:** 1.3.5+Refactored  
            **Author:** Huw Sandaver w/ enhancements and suggestions by ChatGPT  
            **Contact:** hsandaver@alumni.unimelb.edu.au
            
            Enjoy exploring your linked data!
            """
        )

# ------------------------------
# Legends Creation Helper
# ------------------------------
def create_legends(rel_colors: Dict[str, str], node_colors: Dict[str, str]) -> str:
    rel_items = "".join(
        f"<li><span style='color:{color}; font-size: 16px;'>●</span> {rel.replace('_', ' ').title()}</li>"
        for rel, color in rel_colors.items()
    )
    node_items = "".join(
        f"<li><span style='color:{color}; font-size: 16px;'>●</span> {ntype}</li>"
        for ntype, color in node_colors.items()
    )
    return (
        f"<h4>Legends</h4>"
        f"<div style='display:flex;'>"
        f"<ul style='list-style: none; padding: 0; margin-right: 20px;'>"
        f"<strong>Relationships</strong>{rel_items}"
        f"</ul>"
        f"<ul style='list-style: none; padding: 0;'>"
        f"<strong>Node Types</strong>{node_items}"
        f"</ul>"
        f"</div>"
    )

# ------------------------------
# Automated Testing
# ------------------------------
def run_tests():
    import unittest

    class UtilityTests(unittest.TestCase):
        def test_remove_fragment(self):
            uri = "http://example.org/resource#fragment"
            expected = "http://example.org/resource"
            self.assertEqual(remove_fragment(uri), expected)

        def test_normalize_data(self):
            data = {"id": "http://example.org/resource#id", "prefLabel": {"en": "Test Resource"}}
            normalized = normalize_data(data)
            self.assertEqual(normalized["id"], "http://example.org/resource")
            self.assertEqual(normalized["prefLabel"]["en"], "Test Resource")

        def test_is_valid_iiif_manifest(self):
            valid_url = "http://example.org/iiif/manifest.json"
            invalid_url = "ftp://example.org/manifest"
            self.assertTrue(is_valid_iiif_manifest(valid_url))
            self.assertFalse(is_valid_iiif_manifest(invalid_url))

    suite = unittest.TestLoader().loadTestsFromTestCase(UtilityTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print("Some tests failed.")

# ------------------------------
# CI/CD Integration (Instructions)
# ------------------------------
# For CI/CD, include a .github/workflows/ci.yml or similar to run tests & lint.

# ------------------------------
# Entry Point
# ------------------------------
if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_tests()
    else:
        main()
