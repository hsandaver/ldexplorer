import os
import json
import logging
import traceback
from io import StringIO
from urllib.parse import urlparse, urlunparse

import requests
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS
import networkx as nx  # For static layout computation

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
RELATIONSHIP_CONFIG = {
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
    # ▼▼▼ NEW RELATIONSHIPS ▼▼▼
    "sameAs": "#A0522D",
    "child": "#1E90FF",
    "sibling": "#556B2F",
    "spouse": "#CD853F",
    "studentOf": "#8B008B",
    "employedBy": "#B8860B",
    "occupation": "#8FBC8F",
    "fieldOfActivity": "#FF4500"
    # ▲▲▲ NEW RELATIONSHIPS ▲▲▲
}

NODE_TYPE_COLORS = {
    "Person": "#FFA500",
    "Organization": "#87CEEB",
    "Place": "#98FB98",
    "StillImage": "#FFD700",
    "Event": "#DDA0DD",
    "Work": "#20B2AA",
    "Unknown": "#D3D3D3"
}

# Define node shapes for each entity type to further improve aesthetics.
NODE_TYPE_SHAPES = {
    "Person": "circle",
    "Organization": "box",
    "Place": "triangle",
    "StillImage": "dot",  # Fallback shape
    "Event": "star",
    "Work": "ellipse",
    "Unknown": "dot"
}

DEFAULT_NODE_COLOR = "#D3D3D3"
LABEL_CACHE = {}  # Cache for fetched labels

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def remove_fragment(uri: str) -> str:
    """Remove the fragment (e.g., #something) from a URI."""
    try:
        parsed = urlparse(uri)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ''))
    except Exception as e:
        logging.error(f"Error removing fragment from {uri}: {e}")
        return uri

def fetch_label_from_uri(uri: str, cache: dict) -> str:
    """
    Fetch a label from a given URI using RDF parsing.
    Uses a cache to improve performance.
    """
    if uri in cache:
        return cache[uri]

    try:
        g = Graph()
        result = g.parse(uri)
        if result:
            # Attempt to get a preferred label
            label = g.preferredLabel(URIRef(uri))
            if label:
                cache[uri] = str(label[0][1])
                return str(label[0][1])
            # Fallback to any RDFS label
            for _, p, o in g.triples((None, RDFS.label, None)):
                cache[uri] = str(o)
                return str(o)
        cache[uri] = uri
        return uri
    except Exception as e:
        logging.error(f"Error fetching label from {uri}: {e}")
        cache[uri] = uri
        return uri

def normalize_data(data: dict) -> dict:
    """
    Normalize a JSON entity:
      - Remove fragment from 'id'.
      - Ensure 'prefLabel.en' exists.
      - Convert 'type' to a list.
      - Normalize relationships based on RELATIONSHIP_CONFIG.
    """
    if not isinstance(data, dict):
        raise ValueError("Invalid data format. Expected a dictionary.")

    # Normalize ID and labels
    data['id'] = remove_fragment(data.get('id', ''))
    data.setdefault('prefLabel', {})['en'] = data.get('prefLabel', {}).get('en', data['id'])

    # Ensure 'type' is a list
    if 'type' in data:
        data['type'] = data['type'] if isinstance(data['type'], list) else [data['type']]

    # Normalize relationships
    for rel in list(data.keys()):
        if rel not in RELATIONSHIP_CONFIG:
            continue
        values = data[rel]
        normalized_values = []
        # Always work with a list
        if not isinstance(values, list):
            values = [values]
        for value in values:
            normalized_id = None
            if isinstance(value, dict):
                if rel in ["spouse", "studentOf", "employedBy"]:
                    normalized_id = remove_fragment(value.get('carriedOutBy', value.get('id', '')))
                elif rel == 'succeededBy':
                    normalized_id = remove_fragment(value.get('resultedIn', ''))
                elif rel == 'precededBy':
                    normalized_id = remove_fragment(value.get('resultedFrom', ''))
                else:
                    normalized_id = remove_fragment(value.get('id', ''))
            elif isinstance(value, str):
                normalized_id = remove_fragment(value)
            if normalized_id:
                normalized_values.append(normalized_id)
                logging.debug(f"Normalized relationship '{rel}': {data['id']} -> {normalized_id}")
        data[rel] = normalized_values
    return data

def parse_entities(json_files) -> tuple:
    """
    Parse multiple JSON files into normalized entities.
    Returns a tuple: (all_data, id_to_label, errors)
    """
    all_data, id_to_label, errors = [], {}, []

    for file in json_files:
        try:
            file.seek(0)
            json_obj = json.load(file)
            normalized_data = normalize_data(json_obj)
            subject_id = normalized_data['id']
            label = normalized_data['prefLabel']['en']
            entity_types = normalized_data.get('type', ['Unknown'])

            # Generate relationship edges
            edges = [
                (subject_id, related_id, rel)
                for rel in RELATIONSHIP_CONFIG
                for related_id in normalized_data.get(rel, [])
            ]
            logging.info(f"Entity {subject_id}: Added {len(edges)} edges.")

            all_data.append({
                "subject": subject_id,
                "label": label,
                "edges": edges,
                "type": entity_types,
                "metadata": normalized_data
            })
            id_to_label[subject_id] = label
        except Exception as e:
            error_details = f"{getattr(file, 'name', 'Unknown File')}: {str(e)}\n{traceback.format_exc()}"
            errors.append(error_details)
            logging.error(f"Error parsing file: {error_details}")
    return all_data, id_to_label, errors

def sparql_query(query: str, all_data: list) -> set:
    """
    Execute a simplified SPARQL-like query on the dataset.
    Returns a set of node IDs that match the query.
    """
    if not query.strip():
        return set()

    filtered_nodes = set()
    try:
        g = Graph()
        # Bind common namespaces
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)

        # Convert JSON data to RDF triples
        for item in all_data:
            subject_id = item['subject']
            subject = URIRef(subject_id)
            g.add((subject, RDFS.label, Literal(item['label'])))
            # Add custom properties from metadata
            for key, value in item['metadata'].items():
                if key in ["id", "prefLabel", "type"]:
                    continue
                if isinstance(value, list):
                    for val in value:
                        if isinstance(val, str):
                            g.add((subject, URIRef(key), URIRef(val)))
                elif isinstance(value, str):
                    g.add((subject, URIRef(key), URIRef(value)))
            # Add RDF type triples
            for entity_type in item.get('type', []):
                g.add((subject, RDF.type, URIRef(entity_type)))
        # Run the query
        results = g.query(query)
        for row in results:
            for r in row:
                if isinstance(r, URIRef):
                    filtered_nodes.add(str(r))
    except Exception as e:
        logging.error(f"Error executing SPARQL query: {e}")
        st.error(f"Invalid SPARQL query. Please check your syntax: {e}")
    return filtered_nodes

# -----------------------------------------------------------------------------
# Graph Building Helpers
# -----------------------------------------------------------------------------
def add_node(net: Network, node_id: str, label: str, entity_types: list, color: str,
             search_node: str = None, show_labels: bool = True) -> None:
    """
    Helper to add a node to the Pyvis network with enhanced aesthetics.
    Uses the first entity type to determine the node shape.
    Includes widthConstraint to wrap long labels.
    """
    node_title = f"{label}<br>Types: {', '.join(entity_types)}"
    is_search = (search_node and node_id == search_node)
    shape = NODE_TYPE_SHAPES.get(entity_types[0], "dot") if entity_types else "dot"
    net.add_node(
        node_id,
        label=label if show_labels else "",
        title=node_title,
        color=color,
        shape=shape,
        size=20 if is_search else 15,
        font={"size": 12, "face": "Arial", "color": "#343a40"},
        borderWidth=2 if is_search else 1,
        borderColor="#FF5733" if is_search else "#343a40",
        shadow=True,
        widthConstraint={"maximum": 150}  # Limits label width and wraps text
    )
    logging.debug(f"Added node: {label} ({node_id}) with color {color} and shape {shape}")

def add_edge(net: Network, src: str, dst: str, relationship: str, id_to_label: dict,
             search_node: str = None) -> None:
    """Helper to add an edge to the Pyvis network with enhanced styling."""
    edge_color = RELATIONSHIP_CONFIG.get(relationship, "#A9A9A9")
    label_text = " ".join(word.capitalize() for word in relationship.split('_'))
    is_search_edge = search_node and (src == search_node or dst == search_node)
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
    all_data: list,
    id_to_label: dict,
    selected_relationships: list,
    search_node: str = None,
    node_positions: dict = None,
    show_labels: bool = True,
    filtered_nodes: set = None
) -> Network:
    """
    Build and configure the Pyvis graph based on the parsed data.
    Applies dynamic aesthetic adjustments based on the number of nodes.
    """
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#f0f2f6",
        font_color="#343a40"
    )
    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=150,
        spring_strength=0.08
    )

    added_nodes = set()
    edge_set = set()

    # Add nodes from entities
    for item in all_data:
        subject_id = item["subject"]
        if filtered_nodes and subject_id not in filtered_nodes:
            logging.debug(f"Skipping node {subject_id} due to filtering")
            continue
        color = next((NODE_TYPE_COLORS.get(t, DEFAULT_NODE_COLOR) for t in item["type"]), DEFAULT_NODE_COLOR)
        if subject_id not in added_nodes:
            add_node(net, subject_id, id_to_label.get(subject_id, subject_id), item["type"], color,
                     search_node=search_node, show_labels=show_labels)
            added_nodes.add(subject_id)

    # Add nodes and edges for relationships
    for item in all_data:
        for src, dst, relationship in item["edges"]:
            if relationship not in selected_relationships:
                continue
            if filtered_nodes and (src not in filtered_nodes or dst not in filtered_nodes):
                logging.debug(f"Skipping edge {src} --{relationship}--> {dst} due to filtering")
                continue
            if dst not in added_nodes:
                dst_label = id_to_label.get(dst, dst)
                add_node(net, dst, dst_label, ["Unknown"], DEFAULT_NODE_COLOR, show_labels=show_labels)
                added_nodes.add(dst)
            if (src, dst, relationship) not in edge_set:
                add_edge(net, src, dst, relationship, id_to_label, search_node=search_node)
                edge_set.add((src, dst, relationship))

    # Dynamic aesthetic adjustments based on node count
    node_count = len(net.nodes)
    node_font_size = 12 if node_count <= 50 else 10
    edge_font_size = 10 if node_count <= 50 else 8

    net.options = json.loads(f"""
    {{
        "nodes": {{
            "font": {{
                "size": {node_font_size},
                "face": "Arial",
                "color": "#343a40",
                "strokeWidth": 0
            }}
        }},
        "edges": {{
            "font": {{
                "size": {edge_font_size},
                "face": "Arial",
                "align": "middle",
                "color": "#343a40"
            }},
            "smooth": {{
                "type": "continuous"
            }}
        }},
        "physics": {{
            "forceAtlas2Based": {{
                "gravity": -50,
                "centralGravity": 0.01,
                "springLength": 150,
                "springStrength": 0.08
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

def create_legends(relationship_colors: dict, node_type_colors: dict) -> str:
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

def display_node_metadata(node_id: str, all_data: list, id_to_label: dict) -> None:
    """Display metadata for a given node in Streamlit."""
    st.markdown("#### Node Metadata")
    node_metadata = next((item["metadata"] for item in all_data if item["subject"] == node_id), {})

    if node_metadata:
        st.write(f"**Label:** {id_to_label.get(node_id, node_id)}")
        for key, value in node_metadata.items():
            if key == 'prefLabel':
                continue
            st.write(f"**{key}:**")
            if isinstance(value, list):
                for v in value:
                    if isinstance(v, dict):
                        st.write(f"  - {v}")
                    else:
                        if isinstance(v, str) and v.startswith("http"):
                            label = fetch_label_from_uri(v, LABEL_CACHE)
                            st.write(f"  - {label} ({v})")
                        else:
                            st.write(f"  - {v}")
            else:
                if isinstance(value, str) and value.startswith("http"):
                    label = fetch_label_from_uri(value, LABEL_CACHE)
                    st.write(f"  - {label} ({value})")
                else:
                    st.write(f"  - {value}")
    else:
        st.write("No metadata available for this node.")

# -----------------------------------------------------------------------------
# Main Streamlit App
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Linked Data Explorer", page_icon="🕸️", layout="wide")
    st.title("🕸️ Linked Data Network Visualizer")
    st.markdown(
        """
        ### Explore Relationships Between Entities
        Upload multiple JSON files representing entities and generate an interactive network.
        Use the sidebar to filter relationships, search for nodes, manually set node positions,
        and now even edit the graph directly!
        """
    )

    # Initialize session state variables
    state_vars = {
        'node_positions': {},
        'selected_node': None,
        'selected_relationships': list(RELATIONSHIP_CONFIG.keys()),
        'search_term': "",
        'show_labels': True,
        'sparql_query': "",
        'filtered_types': [],
        'enable_physics': True,
        # New graph editing state variables
        'graph_data': None,
        'id_to_label': {}
    }
    for key, default in state_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default

    st.sidebar.header("Controls")
    uploaded_files = st.sidebar.file_uploader(
        label="Upload JSON Files",
        type=["json"],
        accept_multiple_files=True,
        help="Select JSON files describing entities and relationships"
    )

    # Parse files and initialize graph_data if not already done
    if uploaded_files:
        try:
            all_data, id_to_label, errors = parse_entities(uploaded_files)
            # Initialize graph_data for editing only once
            if not st.session_state["graph_data"]:
                st.session_state["graph_data"] = all_data
                st.session_state["id_to_label"] = id_to_label
        except Exception as e:
            st.sidebar.error(f"Error parsing files: {e}")
            st.session_state["graph_data"], st.session_state["id_to_label"] = [], {}
    else:
        st.session_state["graph_data"], st.session_state["id_to_label"] = [], {}

    selected_relationships = st.sidebar.multiselect(
        label="Select Relationship Types to Display",
        options=list(RELATIONSHIP_CONFIG.keys()),
        default=st.session_state.selected_relationships,
        key="selected_relationships_control"
    )
    st.session_state.selected_relationships = selected_relationships

    enable_physics = st.sidebar.checkbox(
        label="Enable Physics Simulation",
        value=st.session_state.enable_physics,
        help="Toggle physics simulation on/off. Off will use a static layout computed with spring layout, nyah~!"
    )
    st.session_state.enable_physics = enable_physics

    if st.sidebar.button("Reset Manual Node Positions"):
        st.session_state.node_positions = {}
        st.sidebar.success("Manual positions have been reset, nyah~!")

    # -------------------------------------------------------------------------
    # New Graph Editing Section
    # -------------------------------------------------------------------------
    st.sidebar.header("✏️ Graph Editing")
    with st.sidebar.expander("Edit Graph"):
        edit_option = st.radio("Select action", ("Add Node", "Delete Node", "Modify Node", "Add Edge", "Delete Edge"))
        
        if edit_option == "Add Node":
            with st.form("add_node_form"):
                new_node_id = st.text_input("Node ID")
                new_node_label = st.text_input("Node Label")
                new_node_type = st.selectbox("Node Type", list(NODE_TYPE_COLORS.keys()))
                submitted = st.form_submit_button("Add Node")
                if submitted and new_node_id and new_node_label:
                    new_node = {
                        "subject": new_node_id,
                        "label": new_node_label,
                        "edges": [],
                        "type": [new_node_type],
                        "metadata": {
                            "id": new_node_id,
                            "prefLabel": {"en": new_node_label},
                            "type": [new_node_type]
                        }
                    }
                    st.session_state.graph_data.append(new_node)
                    st.session_state.id_to_label[new_node_id] = new_node_label
                    st.sidebar.success(f"Node '{new_node_label}' added, nyah~!")
        
        elif edit_option == "Delete Node":
            node_ids = [node["subject"] for node in st.session_state.graph_data]
            if node_ids:
                node_to_delete = st.selectbox("Select Node to Delete", node_ids)
                if st.button("Delete Node"):
                    # Remove the node and remove any edges referencing it
                    st.session_state.graph_data = [node for node in st.session_state.graph_data if node["subject"] != node_to_delete]
                    for node in st.session_state.graph_data:
                        node["edges"] = [edge for edge in node["edges"] if edge[1] != node_to_delete]
                    st.session_state.id_to_label.pop(node_to_delete, None)
                    st.sidebar.success(f"Node '{node_to_delete}' deleted!")
            else:
                st.info("No nodes available to delete, nyah~!")
        
        elif edit_option == "Modify Node":
            node_ids = [node["subject"] for node in st.session_state.graph_data]
            if node_ids:
                node_to_modify = st.selectbox("Select Node to Modify", node_ids)
                node_obj = next((node for node in st.session_state.graph_data if node["subject"] == node_to_modify), None)
                if node_obj:
                    with st.form("modify_node_form"):
                        new_label = st.text_input("New Label", value=node_obj["label"])
                        new_type = st.selectbox("New Type", list(NODE_TYPE_COLORS.keys()),
                                                index=list(NODE_TYPE_COLORS.keys()).index(node_obj["type"][0])
                                                if node_obj["type"][0] in NODE_TYPE_COLORS else 0)
                        submitted = st.form_submit_button("Modify Node")
                        if submitted:
                            node_obj["label"] = new_label
                            node_obj["type"] = [new_type]
                            node_obj["metadata"]["prefLabel"]["en"] = new_label
                            st.session_state.id_to_label[node_to_modify] = new_label
                            st.sidebar.success(f"Node '{node_to_modify}' modified!")
            else:
                st.info("No nodes available to modify, nyah~!")
        
        elif edit_option == "Add Edge":
            if st.session_state.graph_data:
                with st.form("add_edge_form"):
                    source_node = st.selectbox("Source Node", [node["subject"] for node in st.session_state.graph_data])
                    target_node = st.selectbox("Target Node", [node["subject"] for node in st.session_state.graph_data])
                    relationship = st.selectbox("Relationship", list(RELATIONSHIP_CONFIG.keys()))
                    submitted = st.form_submit_button("Add Edge")
                    if submitted:
                        for node in st.session_state.graph_data:
                            if node["subject"] == source_node:
                                node["edges"].append((source_node, target_node, relationship))
                        st.sidebar.success(f"Edge '{relationship}' from '{source_node}' to '{target_node}' added!")
            else:
                st.info("No nodes available to add an edge, nyah~!")
        
        elif edit_option == "Delete Edge":
            # Gather all edges as a list of tuples
            all_edges = []
            for node in st.session_state.graph_data:
                for edge in node["edges"]:
                    all_edges.append(edge)
            if all_edges:
                edge_to_delete = st.selectbox("Select Edge to Delete", all_edges)
                if st.button("Delete Edge"):
                    for node in st.session_state.graph_data:
                        if node["subject"] == edge_to_delete[0]:
                            node["edges"] = [edge for edge in node["edges"] if edge != edge_to_delete]
                    st.sidebar.success("Edge deleted!")
            else:
                st.info("No edges available to delete, nyah~!")
    
    # -------------------------------------------------------------------------
    # Other Controls (Filtering, Search, Manual Positioning)
    # -------------------------------------------------------------------------
    if st.session_state.graph_data:
        all_types = {t for item in st.session_state.graph_data for t in item.get("type", [])}
        filtered_types = st.sidebar.multiselect(
            "Filter by Entity Types",
            options=list(all_types),
            default=st.session_state.filtered_types,
            key="filtered_types_control"
        )
        st.session_state.filtered_types = filtered_types

    search_term = st.sidebar.text_input(
        label="Search for a Node",
        help="Enter the entity name to highlight",
        key="search_term_control",
        value=st.session_state.search_term
    )
    st.session_state.search_term = search_term

    sparql_query_input = st.sidebar.text_area(
        label="SPARQL-like Query",
        help=(
            "Enter a SPARQL-like query to filter nodes. Example:\n"
            "```\nSELECT ?s WHERE {?s rdf:type <http://example.org/Person> .}\n```"
        ),
        key="sparql_query_control",
        value=st.session_state.sparql_query
    )
    st.session_state.sparql_query = sparql_query_input

    show_labels = st.sidebar.checkbox(
        label="Show Node Labels",
        value=st.session_state.show_labels,
        help="Toggle the visibility of node labels to reduce clutter",
        key="show_labels_control"
    )
    st.session_state.show_labels = show_labels

    st.sidebar.markdown(create_legends(RELATIONSHIP_CONFIG, NODE_TYPE_COLORS), unsafe_allow_html=True)

    st.sidebar.header("📍 Manual Node Positioning")
    if st.session_state.graph_data:
        unique_nodes = {entity['subject']: entity['label'] for entity in st.session_state.graph_data}
        for entity in st.session_state.graph_data:
            for _, dst, _ in entity['edges']:
                unique_nodes.setdefault(dst, st.session_state.id_to_label.get(dst, dst))

        selected_node = st.sidebar.selectbox(
            label="Select a Node to Position",
            options=list(unique_nodes.keys()),
            format_func=lambda x: unique_nodes.get(x, x),
            key="selected_node_control"
        )
        st.session_state.selected_node = selected_node
        if selected_node:
            with st.sidebar.form(key="position_form"):
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
                    st.sidebar.success(
                        f"Position for '{unique_nodes[selected_node]}' set to (X: {x_pos}, Y: {y_pos})"
                    )

    graph_html = None
    if st.session_state.graph_data:
        # Display parsing errors if any (from file upload)
        if uploaded_files:
            all_data, _, errors = parse_entities(uploaded_files)
            if errors:
                with st.expander("⚠️ Parsing Errors"):
                    for error in errors:
                        st.error(error)
        
        if st.session_state.graph_data:
            search_node_id = None
            if search_term.strip():
                for entity in st.session_state.graph_data:
                    if entity['label'].lower() == search_term.lower():
                        search_node_id = entity['subject']
                        break
                if not search_node_id:
                    for node_id, label in st.session_state.id_to_label.items():
                        if label.lower() == search_term.lower():
                            search_node_id = node_id
                            break
                if not search_node_id:
                    st.sidebar.warning("Node not found. Please check the name and try again.")

            filtered_nodes = sparql_query(sparql_query_input, st.session_state.graph_data)
            if st.session_state.filtered_types:
                filtered_by_type = {item["subject"] for item in st.session_state.graph_data
                                    if any(t in st.session_state.filtered_types for t in item.get("type", []))}
                filtered_nodes = filtered_nodes.intersection(filtered_by_type) if filtered_nodes else filtered_by_type

            node_positions = st.session_state.node_positions or None

            with st.spinner("Generating Network Graph..."):
                net = build_graph(
                    all_data=st.session_state.graph_data,
                    id_to_label=st.session_state.id_to_label,
                    selected_relationships=st.session_state.selected_relationships,
                    search_node=search_node_id,
                    node_positions=node_positions,
                    show_labels=show_labels,
                    filtered_nodes=filtered_nodes
                )

            if enable_physics:
                for node in net.nodes:
                    pos = st.session_state.node_positions.get(node.get("id"))
                    if pos:
                        node['x'] = pos['x']
                        node['y'] = pos['y']
                        node['fixed'] = True
                        node['physics'] = False
            else:
                net.options = json.loads("""
                {
                    "nodes": {
                        "font": {
                            "size": 12,
                            "face": "Arial",
                            "color": "#343a40",
                            "strokeWidth": 0
                        }
                    },
                    "edges": {
                        "font": {
                            "size": 10,
                            "face": "Arial",
                            "align": "middle",
                            "color": "#343a40"
                        },
                        "smooth": {
                            "type": "continuous"
                        }
                    },
                    "physics": {"enabled": false},
                    "interaction": {
                        "hover": true,
                        "navigationButtons": true,
                        "zoomView": true,
                        "dragNodes": false,
                        "multiselect": true,
                        "selectConnectedEdges": true
                    }
                }
                """)
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

            if len(net.nodes) > 50 and show_labels:
                st.info("The graph has many nodes, nyah~! Consider toggling 'Show Node Labels' off for better readability.")

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
                st.info("Click on a node to display its metadata.")

            st.markdown("### 📥 Export Options")
            col1, col2, col3 = st.columns(3)
            with col1:
                if graph_html:
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
                    st.warning(
                        "Exporting as PNG is not supported directly. Please export as HTML and take a screenshot."
                    )
            with col3:
                if st.button("Download Graph Data as JSON"):
                    graph_data = {
                        "nodes": [
                            {"id": node['id'], "label": node['label'], "metadata": node.get('title', ''),
                             "x": node.get('x'), "y": node.get('y')}
                            for node in net.nodes
                        ],
                        "edges": [
                            {"from": edge['from'], "to": edge['to'], "label": edge['label'],
                             "metadata": edge.get('title', '')}
                            for edge in net.edges
                        ]
                    }
                    graph_json = json.dumps(graph_data, indent=2)
                    st.download_button(
                        label="Download Graph Data as JSON",
                        data=graph_json,
                        file_name="graph_data.json",
                        mime="application/json"
                    )
        else:
            st.warning("No valid data found. Please check your JSON files.")
    else:
        st.info("🗂️ Upload JSON files containing linked data entities in the sidebar.")

    st.markdown(
        """
        <style>
            .stApp {
                max-width: 1600px;
                padding: 1rem;
            }
            h1, h3, h4 {
                color: #333;
            }
            .stButton > button, .stDownloadButton > button {
                background-color: #5cb85c;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                cursor: pointer;
                border-radius: 0.25rem;
            }
            .stButton > button:hover, .stDownloadButton > button:hover {
                background-color: #4cae4c;
            }
        </style>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
