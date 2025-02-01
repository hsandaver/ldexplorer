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
    """Helper to add a node to the Pyvis network."""
    node_title = f"{label}<br>Types: {', '.join(entity_types)}"
    is_search = (search_node and node_id == search_node)
    net.add_node(
        node_id,
        label=label if show_labels else "",
        title=node_title,
        color=color,
        shape="dot",
        size=20 if is_search else 15,
        font={"size": 12, "face": "Arial", "color": "#343a40"},
        borderWidth=2 if is_search else 1,
        borderColor="#FF5733" if is_search else "#343a40"
    )
    logging.debug(f"Added node: {label} ({node_id}) with color {color}")


def add_edge(net: Network, src: str, dst: str, relationship: str, id_to_label: dict,
             search_node: str = None) -> None:
    """Helper to add an edge to the Pyvis network."""
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
        # Filter out nodes if needed
        if filtered_nodes and subject_id not in filtered_nodes:
            logging.debug(f"Skipping node {subject_id} due to filtering")
            continue

        # Get color based on entity type
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
            # Ensure destination node exists
            if dst not in added_nodes:
                dst_label = id_to_label.get(dst, dst)
                add_node(net, dst, dst_label, ["Unknown"], DEFAULT_NODE_COLOR, show_labels=show_labels)
                added_nodes.add(dst)
            if (src, dst, relationship) not in edge_set:
                add_edge(net, src, dst, relationship, id_to_label, search_node=search_node)
                edge_set.add((src, dst, relationship))

    # Global Pyvis graph options
    net.set_options("""
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
        "physics": {
            "forceAtlas2Based": {
                "gravity": -50,
                "centralGravity": 0.01,
                "springLength": 150,
                "springStrength": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        },
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "zoomView": true,
            "dragNodes": true,
            "multiselect": true,
            "selectConnectedEdges": true
        }
    }
    """)
    return net


def create_legend(relationship_colors: dict) -> str:
    """Generate an HTML legend for relationship types."""
    legend_items = [
        f"<li><span style='color:{color};'>●</span> {rel.replace('_', ' ').title()}</li>"
        for rel, color in relationship_colors.items()
    ]
    return f"<h4>Legend</h4><ul style='list-style: none; padding: 0;'>{''.join(legend_items)}</ul>"


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
        Use the sidebar to filter relationships, search for nodes, and manually set node positions.
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
        'filtered_types': []
    }
    for key, default in state_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # -----------------------------------------------------------------------------
    # Sidebar Controls
    # -----------------------------------------------------------------------------
    st.sidebar.header("Controls")
    uploaded_files = st.sidebar.file_uploader(
        label="Upload JSON Files",
        type=["json"],
        accept_multiple_files=True,
        help="Select JSON files describing entities and relationships"
    )

    selected_relationships = st.sidebar.multiselect(
        label="Select Relationship Types to Display",
        options=list(RELATIONSHIP_CONFIG.keys()),
        default=st.session_state.selected_relationships,
        key="selected_relationships_control"
    )
    st.session_state.selected_relationships = selected_relationships

    # Parse entities if files are uploaded
    if uploaded_files:
        try:
            all_data, id_to_label, errors = parse_entities(uploaded_files)
        except Exception as e:
            st.sidebar.error(f"Error parsing files: {e}")
            all_data, id_to_label, errors = [], {}, [str(e)]
    else:
        all_data, id_to_label, errors = [], {}, []

    if all_data:
        # Filter by Entity Types
        all_types = {t for item in all_data for t in item.get("type", [])}
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

    st.sidebar.markdown(create_legend(RELATIONSHIP_CONFIG), unsafe_allow_html=True)

    # Manual Node Positioning
    st.sidebar.header("📍 Manual Node Positioning")
    if uploaded_files and all_data:
        unique_nodes = {entity['subject']: entity['label'] for entity in all_data}
        # Include nodes from edges as well
        for entity in all_data:
            for _, dst, _ in entity['edges']:
                unique_nodes.setdefault(dst, id_to_label.get(dst, dst))

        selected_node = st.sidebar.selectbox(
            label="Select a Node to Position",
            options=list(unique_nodes.keys()),
            format_func=lambda x: unique_nodes.get(x, x),
            key="selected_node_control"
        )
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

    # -----------------------------------------------------------------------------
    # Main Display
    # -----------------------------------------------------------------------------
    if uploaded_files:
        if errors:
            with st.expander("⚠️ Parsing Errors"):
                for error in errors:
                    st.error(error)
        if all_data:
            # Try to find a matching node for search term
            search_node_id = None
            if search_term.strip():
                for entity in all_data:
                    if entity['label'].lower() == search_term.lower():
                        search_node_id = entity['subject']
                        break
                if not search_node_id:
                    for node_id, label in id_to_label.items():
                        if label.lower() == search_term.lower():
                            search_node_id = node_id
                            break
                if not search_node_id:
                    st.sidebar.warning("Node not found. Please check the name and try again.")

            # Apply SPARQL query filtering
            filtered_nodes = sparql_query(sparql_query_input, all_data)
            if st.session_state.filtered_types:
                filtered_by_type = {item["subject"] for item in all_data
                                    if any(t in st.session_state.filtered_types for t in item.get("type", []))}
                filtered_nodes = filtered_nodes.intersection(filtered_by_type) if filtered_nodes else filtered_by_type

            node_positions = st.session_state.node_positions or None

            with st.spinner("Generating Network Graph..."):
                net = build_graph(
                    all_data=all_data,
                    id_to_label=id_to_label,
                    selected_relationships=st.session_state.selected_relationships,
                    search_node=search_node_id,
                    node_positions=node_positions,
                    show_labels=show_labels,
                    filtered_nodes=filtered_nodes
                )

            # Apply fixed positions if available
            for node in net.nodes:
                pos = st.session_state.node_positions.get(node.get("id"))
                if pos:
                    node['x'] = pos['x']
                    node['y'] = pos['y']
                    node['fixed'] = True
                    node['physics'] = False
            if not show_labels:
                for node in net.nodes:
                    node['label'] = ""

            try:
                output_path = "network_graph.html"
                net.save_graph(output_path)
                with open(output_path, "r", encoding="utf-8") as f:
                    graph_html = f.read()
                components.html(graph_html, height=750, scrolling=True)
                os.remove(output_path)
            except Exception as e:
                st.error(f"Graph generation failed: {e}")

            # Display stats and node metadata
            st.markdown(f"**Total Nodes:** {len(net.nodes)} | **Total Edges:** {len(net.edges)}")
            if st.session_state.selected_node:
                display_node_metadata(st.session_state.selected_node, all_data, id_to_label)
            else:
                st.markdown("#### Node Metadata")
                st.info("Click on a node to display its metadata.")

            # Export Options
            st.markdown("### 📥 Export Options")
            col1, col2, col3 = st.columns(3)
            with col1:
                try:
                    with open("network_graph.html", "r", encoding="utf-8") as f:
                        graph_content = f.read()
                    st.download_button(
                        label="Download Graph as HTML",
                        data=graph_content,
                        file_name="network_graph.html",
                        mime="text/html"
                    )
                except Exception:
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

    # -----------------------------------------------------------------------------
    # Additional CSS Styling
    # -----------------------------------------------------------------------------
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
