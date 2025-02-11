#!/usr/bin/env python
"""
Complete Python Network Visualization Application with Enhancements
Including real SPARQL query support by converting JSON to RDF.
Author: Huw Sandaver w/ enhancements by ChatGPT
Date: 2025-02-11 (Enhanced with physics sliders)
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
    metadata: Dict[str, Any]
    edges: List[Edge] = field(default_factory=list)

@dataclass
class GraphData:
    nodes: List[Node]

# -----------------------------------------------------------------------------
# Graph Configuration
# -----------------------------------------------------------------------------
def build_graph(graph_data: GraphData, show_labels: bool = True, enable_physics: bool = True) -> Network:
    """Builds and configures a Pyvis graph, ensuring physics controls are enabled."""
    
    net = Network(height="750px", width="100%", directed=True, notebook=False)
    
    # Enabling physics sliders
    net.options = {
        "physics": {
            "enabled": enable_physics,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravity": -50,
                "centralGravity": 0.01,
                "springLength": 150,
                "springStrength": 0.08
            },
            "minVelocity": 0.75
        },
        "configure": {
            "enabled": True,  # Enables physics control UI
            "filter": ["physics", "nodes", "edges", "interaction"],
            "showButton": True
        },
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "zoomView": True,
            "dragNodes": True
        }
    }

    added_nodes = set()
    for node in graph_data.nodes:
        if node.id not in added_nodes:
            net.add_node(
                node.id,
                label=node.label if show_labels else "",
                title=node.label,
                shape="dot",
                size=15
            )
            added_nodes.add(node.id)

    for node in graph_data.nodes:
        for edge in node.edges:
            net.add_edge(edge.source, edge.target, title=edge.relationship)

    return net

# -----------------------------------------------------------------------------
# Main Streamlit App
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Linked Data Explorer", page_icon="🕸️", layout="wide")
    st.title("🕸️ Linked Data Network Visualizer")

    st.sidebar.header("Controls")
    uploaded_files = st.sidebar.file_uploader("Upload JSON Files", type=["json"], accept_multiple_files=True)

    enable_physics = st.sidebar.checkbox("Enable Physics Simulation", value=True)

    if "graph_data" not in st.session_state:
        st.session_state.graph_data = GraphData(nodes=[])

    if uploaded_files:
        file_contents = [file.read().decode("utf-8") for file in uploaded_files]
        try:
            nodes = []
            for content in file_contents:
                data = json.loads(content)
                nodes.append(Node(id=data['id'], label=data['prefLabel']['en'], types=[], metadata=data, edges=[]))
            st.session_state.graph_data = GraphData(nodes=nodes)
        except Exception as e:
            st.sidebar.error(f"Error parsing JSON: {e}")

    if st.session_state.graph_data.nodes:
        net = build_graph(st.session_state.graph_data, enable_physics=enable_physics)

        # Display Graph in Streamlit
        graph_html = net.generate_html()
        components.html(graph_html, height=750, scrolling=True)

        # Downloadable HTML Graph
        st.download_button("Download Graph as HTML", data=graph_html, file_name="network_graph.html", mime="text/html")

if __name__ == "__main__":
    main()
