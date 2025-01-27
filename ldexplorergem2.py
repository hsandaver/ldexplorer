import streamlit as st
import streamlit.components.v1 as components
import json
from pyvis.network import Network
import os
import logging
from urllib.parse import urlparse, urlunparse
import traceback
import requests
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS
from io import StringIO


# Configure logging
logging.basicConfig(level=logging.INFO)

#################################################################
# Relationship Colors Configuration
#################################################################

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

#################################################################
# Node Type Colors Configuration
#################################################################

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

#################################################################
# URI Fragment Removal
#################################################################

def remove_fragment(uri: str) -> str:
    """
    Remove the fragment (e.g., #something) from a URI.
    Returns the cleaned URI or the original URI if parsing fails.
    """
    try:
        parsed = urlparse(uri)
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            ''
        ))
    except Exception as e:
        logging.error(f"Error removing fragment from {uri}: {e}")
        return uri

#################################################################
# Label Fetching
#################################################################

LABEL_CACHE = {} # Cache to avoid repetitive requests

def fetch_label_from_uri(uri: str, cache: dict) -> str:
    """
    Fetches a label from a given URI. Uses a cache to improve performance.
    Returns the label (if found) or the original URI if lookup fails.
    """
    if uri in cache:
        return cache[uri]

    try:
        g = Graph()
        result = g.parse(uri)
        
        if result:
            
            label = g.preferredLabel(URIRef(uri))
            if label:
                cache[uri] = str(label[0][1])
                return str(label[0][1])
            else:
              
               for s,p,o in g:
                  if p == RDFS.label:
                      cache[uri] = str(o)
                      return str(o)
            
            cache[uri] = uri # If no label, store the uri
            return uri


        else:
            cache[uri] = uri
            return uri

    except Exception as e:
        logging.error(f"Error fetching label from {uri}: {e}")
        cache[uri] = uri # Store original URI
        return uri


#################################################################
# Data Normalization
#################################################################

def normalize_data(data: dict) -> dict:
    """
    Normalize a single JSON entity's structure:
      - Convert 'id' to a fragment-free form.
      - Ensure 'prefLabel.en' always exists.
      - Convert 'type' to a list if needed.
      - Normalize relationships according to RELATIONSHIP_CONFIG.
    """
    if not isinstance(data, dict):
        raise ValueError("Invalid data format. Expected a dictionary.")

    # Normalize ID
    data['id'] = remove_fragment(data.get('id', ''))

    # Normalize labels with fallback
    data['prefLabel'] = data.get('prefLabel', {})
    data['prefLabel']['en'] = data['prefLabel'].get('en', data['id'])

    # Normalize type to list
    if 'type' in data:
        data['type'] = [data['type']] if isinstance(data['type'], str) else data['type']

    # Normalize relationships
    for rel in list(data.keys()):
        # Only process if it's in our RELATIONSHIP_CONFIG
        if rel in RELATIONSHIP_CONFIG:
            values = data[rel]
            normalized_values = []

            # Ensure values are always a list
            if not isinstance(values, list):
                values = [values]

            for value in values:
                # If the relationship is spouse, studentOf, or employedBy,
                # they might use "carriedOutBy" for the real ID
                if rel in ["spouse", "studentOf", "employedBy"] and isinstance(value, dict):
                    carried_out_by = value.get('carriedOutBy')
                    if carried_out_by:
                        normalized_id = remove_fragment(carried_out_by)
                        normalized_values.append(normalized_id)
                        logging.debug(
                            f"Relationship '{rel}': '{data['id']}' -> '{normalized_id}' (carriedOutBy)"
                        )
                    else:
                        fallback_id = value.get('id')
                        if fallback_id:
                            normalized_id = remove_fragment(fallback_id)
                            normalized_values.append(normalized_id)
                            logging.debug(
                                f"Relationship '{rel}' fallback: "
                                f"'{data['id']}' -> '{normalized_id}'"
                            )

                # Handle succeededBy/precededBy with special fields
                elif isinstance(value, dict):
                    if rel == 'succeededBy':
                        related_id = value.get('resultedIn')
                    elif rel == 'precededBy':
                        related_id = value.get('resultedFrom')
                    else:
                        related_id = value.get('id')

                    if related_id:
                        normalized_id = remove_fragment(related_id)
                        normalized_values.append(normalized_id)
                        logging.debug(
                            f"Relationship '{rel}': '{data['id']}' -> '{normalized_id}'"
                        )

                # Otherwise, if it's a simple string (like sameAs, child, sibling, etc.)
                else:
                    normalized_id = remove_fragment(value)
                    normalized_values.append(normalized_id)
                    logging.debug(
                        f"Relationship '{rel}': '{data['id']}' -> '{normalized_id}' (simple string)"
                    )

            data[rel] = normalized_values

    return data


#################################################################
# Entities Parsing
#################################################################

def parse_entities(json_files) -> tuple:
    """
    Parse multiple JSON files with robust error handling.
    Returns:
        all_data (list): A list of dicts with normalized entity info.
        id_to_label (dict): Maps entity IDs to their 'en' label.
        errors (list): A list of error messages (if any).
    """
    all_data = []
    id_to_label = {}
    errors = []

    for file in json_files:
        try:
            file.seek(0)
            json_obj = json.load(file)
            normalized_data = normalize_data(json_obj)

            subject_id = normalized_data['id']
            label = normalized_data['prefLabel']['en']
            entity_types = normalized_data.get('type', ['Unknown'])

            # Generate edges
            edges = []
            for relationship in RELATIONSHIP_CONFIG:
                for related_id in normalized_data.get(relationship, []):
                    edges.append((subject_id, related_id, relationship))
                    logging.info(
                        f"Adding edge: {subject_id} --{relationship}--> {related_id}"
                    )

            # Store parsed entity data
            all_data.append({
                "subject": subject_id,
                "label": label,
                "edges": edges,
                "type": entity_types,
                "metadata": normalized_data
            })

            id_to_label[subject_id] = label

        except Exception as e:
            error_details = f"{file.name}: {str(e)}\n{traceback.format_exc()}"
            errors.append(error_details)
            logging.error(f"Error parsing file {file.name}: {e}")

    return all_data, id_to_label, errors

#################################################################
# SPARQL Query Functionality
#################################################################

def sparql_query(query: str, all_data: list) -> set:
    """
    Executes a simplified SPARQL-like query on the dataset.
    This function only handles simple triple patterns for now.
    Returns a set of node IDs that match the query.
    """
    if not query:
        return set()

    filtered_nodes = set()

    try:
        g = Graph()
        # Define custom namespaces if you have them
        ns_map = {
            "rdf": RDF,
            "rdfs": RDFS,
        }
        for prefix, ns in ns_map.items():
             g.bind(prefix, ns)

        # Convert JSON data to RDF triples for querying
        for item in all_data:
            subject_id = item['subject']
            subject = URIRef(subject_id)
            g.add((subject, RDFS.label, Literal(item['label'])))
            for k, v in item['metadata'].items():
                if k != "id" and k != "prefLabel" and k != 'type':
                    if isinstance(v, list):
                       for val in v:
                            if isinstance(val, str):
                                  g.add((subject, URIRef(k), URIRef(val)))
                    else:
                       if isinstance(v, str):
                            g.add((subject, URIRef(k), URIRef(v)))

            for entity_type in item.get('type', []):
              g.add((subject, RDF.type, URIRef(entity_type)))

        results = g.query(query)

        if results:
            for row in results:
                for r in row:
                   if isinstance(r, URIRef):
                     filtered_nodes.add(str(r))

    except Exception as e:
        logging.error(f"Error executing SPARQL query: {e}")
        st.error(f"Invalid SPARQL query. Please check your syntax: {e}")

    return filtered_nodes


#################################################################
# Graph Building
#################################################################

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
    Arguments:
        all_data: Parsed entity data from JSON files.
        id_to_label: Mapping from IDs to labels.
        selected_relationships: Relationship types to display.
        search_node: Optional node to highlight.
        node_positions: Optional dict with manual x,y positions for nodes.
        show_labels: Whether to display node labels on the graph.
    Returns:
        net (Network): Configured Pyvis network object.
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

    # Add nodes
    for item in all_data:
        subject_id = item["subject"]
        label = id_to_label.get(subject_id, subject_id)
        entity_types = item["type"]
        color = next(
            (NODE_TYPE_COLORS.get(t, DEFAULT_NODE_COLOR) for t in entity_types),
            DEFAULT_NODE_COLOR
        )

        # Node Filtering
        if filtered_nodes and subject_id not in filtered_nodes:
            logging.debug(f"Skipping node: {label} ({subject_id}) - Filtered")
            continue

        if subject_id not in added_nodes:
            node_title = f"{label}<br>Types: {', '.join(entity_types)}"
            node_size = 20 if (search_node and subject_id == search_node) else 15
            node_color = color
            border_width = 2 if (search_node and subject_id == search_node) else 1
            border_color = "#FF5733" if (search_node and subject_id == search_node) else "#343a40"

            net.add_node(
                subject_id,
                label=label if show_labels else "",
                title=node_title,
                color=node_color,
                shape="dot",
                size=node_size,
                font={"size": 12, "face": "Arial", "color": "#343a40"},
                borderWidth=border_width,
                borderColor=border_color
            )
            added_nodes.add(subject_id)
            logging.debug(f"Added node: {label} ({subject_id}) with color {node_color}")

    # Add edges
    for item in all_data:
        for src, dst, relationship in item["edges"]:
            if relationship not in selected_relationships:
                continue

             # Apply node filter to edges
            if filtered_nodes:
                if src not in filtered_nodes or dst not in filtered_nodes:
                     logging.debug(f"Skipping edge: {src} --{relationship}--> {dst} - Filtered")
                     continue
            # Ensure destination node exists
            if dst not in added_nodes:
                dst_label = id_to_label.get(dst, dst)
                net.add_node(
                    dst,
                    label=dst_label if show_labels else "",
                    title=dst_label,
                    color=DEFAULT_NODE_COLOR,
                    shape="dot",
                    size=15,
                    font={"size": 12, "face": "Arial", "color": "#343a40"},
                    borderWidth=1,
                    borderColor="#343a40"
                )
                added_nodes.add(dst)
                logging.info(f"Added node: {dst_label} ({dst}) with default color")

            # Skip duplicates
            if (src, dst, relationship) in edge_set:
                logging.debug(f"Skipping duplicate edge: {src} --{relationship}--> {dst}")
                continue

            edge_color = RELATIONSHIP_CONFIG.get(relationship, "#A9A9A9")
            label_text = " ".join(word.capitalize() for word in relationship.split('_'))

            # Highlight edges if connected to the searched node
            if search_node and (src == search_node or dst == search_node):
                edge_width = 3
                edge_color = "#FF5733"
            else:
                edge_width = 2

            net.add_edge(
                src,
                dst,
                label=label_text,
                color=edge_color,
                width=edge_width,
                arrows='to',
                title=f"{label_text}: {id_to_label.get(src, src)} → {id_to_label.get(dst, dst)}",
                font={"size": 10, "align": "middle"},
                smooth={'enabled': True, 'type': 'continuous'}
            )
            logging.debug(f"Edge added: {src} --{label_text}--> {dst} with color {edge_color}")
            edge_set.add((src, dst, relationship))

    # Additional global styles
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


#################################################################
# Legend Creation
#################################################################

def create_legend(relationship_colors: dict) -> str:
    """
    Create an HTML-based legend for relationship types and their colors.
    """
    legend_html = "<h4>Legend</h4><ul style='list-style: none; padding: 0;'>"
    for rel, color in relationship_colors.items():
        rel_name = rel.replace('_', ' ').title()
        legend_html += f"<li><span style='color:{color};'>●</span> {rel_name}</li>"
    legend_html += "</ul>"
    return legend_html

#################################################################
# Display Node Metadata
#################################################################

def display_node_metadata(node_id: str, all_data: list, id_to_label: dict):
    """
    Displays a metadata panel for a selected node, including all available properties and labels in all languages.
    """
    st.markdown("#### Node Metadata")

    node_metadata = {}
    for item in all_data:
        if item['subject'] == node_id:
            node_metadata = item['metadata']
            break

    if node_metadata:
        st.write(f"**Label:** {id_to_label.get(node_id, node_id)}")
        for key, value in node_metadata.items():
            if key != 'prefLabel':
             st.write(f"**{key}:**")
             if isinstance(value, list):
                for v in value:
                    if isinstance(v, dict):
                        st.write(f"  - {v}")
                    else:
                        # Attempt to fetch labels for values that look like URIs
                        if str(v).startswith("http"):
                            label = fetch_label_from_uri(v, LABEL_CACHE)
                            st.write(f"  - {label} ({v})")
                        else:
                            st.write(f"  - {v}")
             else:
                # Attempt to fetch labels for values that look like URIs
                 if isinstance(value, str) and value.startswith("http"):
                    label = fetch_label_from_uri(value, LABEL_CACHE)
                    st.write(f"  - {label} ({value})")
                 else:
                    st.write(f"  - {value}")
    else:
        st.write("No metadata available for this node.")


#################################################################
# Main App
#################################################################

def main():
    """
    Streamlit application for uploading JSON files, parsing them into
    a network graph, and displaying the results via Pyvis.
    """
    st.set_page_config(page_title="Linked Data Explorer", page_icon="🕸️", layout="wide")
    st.title("🕸️ Linked Data Network Visualizer")

    st.markdown("""
    ### Explore Relationships Between Entities
    Upload multiple JSON files that represent entities, and generate an
    interactive relationship network. Use the sidebar to filter relationships,
    search for specific entities, and manually set node positions to reduce clutter.
    """)
    st.markdown("""
    #### Tips
     - **SPARQL queries:** use the *graph pattern matching* to filter the graph
     - **Node details:** Click a node to display details below the graph
     - **Persistent positions**: Node positions are preserved between sessions

    """)


    # Initialize session state
    if 'node_positions' not in st.session_state:
        st.session_state.node_positions = {}
    if 'selected_node' not in st.session_state:
      st.session_state.selected_node = None
    if 'selected_relationships' not in st.session_state:
        st.session_state.selected_relationships = list(RELATIONSHIP_CONFIG.keys())
    if 'search_term' not in st.session_state:
        st.session_state.search_term = ""
    if 'show_labels' not in st.session_state:
        st.session_state.show_labels = True
    if 'sparql_query' not in st.session_state:
         st.session_state.sparql_query = ""
    if 'filtered_types' not in st.session_state:
         st.session_state.filtered_types = []

    ########################
    # Sidebar Controls
    ########################
    st.sidebar.header("Controls")

    # Upload JSON files
    uploaded_files = st.sidebar.file_uploader(
        label="Upload JSON Files",
        type=["json"],
        accept_multiple_files=True,
        help="Select multiple JSON files describing entities and their relationships"
    )

    # Relationship filter
    relationship_types = list(RELATIONSHIP_CONFIG.keys())
    selected_relationships = st.sidebar.multiselect(
        label="Select Relationship Types to Display",
        options=relationship_types,
        default=st.session_state.selected_relationships,
        key = "selected_relationships_control"
    )

    st.session_state.selected_relationships = selected_relationships

   # Entity Type filter
    if uploaded_files:
            try:
                all_data, id_to_label, errors = parse_entities(uploaded_files)
            except Exception as e:
                st.sidebar.error(f"Error parsing files: {e}")
                all_data, id_to_label, errors = [], {}, [str(e)]
    else:
        all_data, id_to_label, errors = [], {}, []

    if all_data:
        all_types = set()
        for item in all_data:
            for type in item.get("type", []):
                 all_types.add(type)
        filtered_types = st.sidebar.multiselect(
            "Filter by Entity Types",
            options = list(all_types),
            default=st.session_state.filtered_types,
             key="filtered_types_control"
        )
        st.session_state.filtered_types = filtered_types



    # Node search
    search_term = st.sidebar.text_input(
        label="Search for a Node",
        help="Enter the name of the entity to highlight",
        key="search_term_control",
        value = st.session_state.search_term
    )

    st.session_state.search_term = search_term

    # SPARQL Query
    sparql_query_input = st.sidebar.text_area(
        label="SPARQL-like Query",
        help="""
         Enter a SPARQL-like query to filter nodes. Example:
         ```
         SELECT ?s WHERE {?s rdf:type <http://example.org/Person> .}
         ```
          For specific properties use the URI, e.g., `<http://example.org/placeOfBirth>`.
         """,
        key = "sparql_query_control",
         value = st.session_state.sparql_query
    )

    st.session_state.sparql_query = sparql_query_input


    # Toggle node labels
    show_labels = st.sidebar.checkbox(
        label="Show Node Labels",
        value=st.session_state.show_labels,
        help="Toggle the visibility of node labels to reduce clutter",
        key="show_labels_control"
    )

    st.session_state.show_labels = show_labels

    # Legend
    st.sidebar.markdown(create_legend(RELATIONSHIP_CONFIG), unsafe_allow_html=True)

    ########################
    # Manual Node Positioning
    ########################
    st.sidebar.header("📍 Manual Node Positioning")

    if uploaded_files and all_data:
        # Collect all unique node labels
        unique_nodes = {entity['subject']: entity['label'] for entity in all_data}
        for entity in all_data:
            for (_, dst, _) in entity['edges']:
                unique_nodes[dst] = id_to_label.get(dst, dst)

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
                submit_button = st.form_submit_button(label="Set Position")

                if submit_button:
                    st.session_state.node_positions[selected_node] = {"x": x_pos, "y": y_pos}
                    st.sidebar.success(
                        f"Position for '{unique_nodes[selected_node]}' set to (X: {x_pos}, Y: {y_pos})"
                    )

    ########################
    # Main Display
    ########################
    if uploaded_files:
        if errors:
            with st.expander("⚠️ Parsing Errors"):
                for error in errors:
                    st.error(error)

        if all_data:

            # Attempt to find the node ID for the search term
            search_node_id = None
            if search_term:
                # Look among 'subject' labels
                for entity in all_data:
                    if entity['label'].lower() == search_term.lower():
                        search_node_id = entity['subject']
                        break

                # If not found, check the id_to_label mapping
                if not search_node_id:
                    for node_id, label in id_to_label.items():
                        if label.lower() == search_term.lower():
                            search_node_id = node_id
                            break

                if not search_node_id and search_term.strip():
                    st.sidebar.warning("Node not found. Please check the name and try again.")

             # Apply SPARQL query
            filtered_nodes = sparql_query(sparql_query_input, all_data)

            # Apply entity type filtering
            if filtered_types:
                filtered_nodes_by_type = set()
                for item in all_data:
                    for t in item.get("type", []):
                       if t in filtered_types:
                         filtered_nodes_by_type.add(item["subject"])
                         break
                if filtered_nodes:
                  filtered_nodes = filtered_nodes.intersection(filtered_nodes_by_type)
                else:
                  filtered_nodes = filtered_nodes_by_type


            # Prepare node positions
            node_positions = st.session_state.node_positions if st.session_state.node_positions else None

            with st.spinner("Generating Network Graph..."):
                net = build_graph(
                    all_data=all_data,
                    id_to_label=id_to_label,
                    selected_relationships=selected_relationships,
                    search_node=search_node_id,
                    node_positions=node_positions,
                    show_labels=show_labels,
                    filtered_nodes=filtered_nodes
                )

            # Apply fixed positions if available
            for node_id, pos in st.session_state.node_positions.items():
                if node_id in net.node_ids:
                    net.nodes[node_id]['x'] = pos['x']
                    net.nodes[node_id]['y'] = pos['y']
                    net.nodes[node_id]['fixed'] = True
                    net.nodes[node_id]['physics'] = False

            # Hide labels if toggled off
            if not show_labels:
                for node in net.nodes:
                    node['label'] = ""

            # Generate and display the network graph
            try:
                output_path = "network_graph.html"
                net.save_graph(output_path)

                with open(output_path, "r", encoding="utf-8") as f:
                    graph_html = f.read()
                components.html(graph_html, height=750, scrolling=True)
                os.remove(output_path)

            except Exception as e:
                st.error(f"Graph generation failed: {e}")

            # Display basic stats
            total_nodes = len(net.nodes)
            total_edges = len(net.edges)
            st.markdown(f"**Total Nodes:** {total_nodes} | **Total Edges:** {total_edges}")

            # Metadata Display for selected node
            if selected_node:
               display_node_metadata(selected_node, all_data, id_to_label)
            else:
               st.markdown("#### Node Metadata")
               st.info("Click on a node to display its metadata.")

            # Export section
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
                        "Exporting as PNG is not supported directly. "
                        "Please export as HTML and take a screenshot."
                    )

            with col3:
                # Download graph data as JSON
                if st.button("Download Graph Data as JSON"):
                      graph_data = {
                          "nodes": [{"id": node['id'], "label": node['label'], "metadata": node.get('title',''), "x":node.get('x'), "y":node.get('y') } for node in net.nodes],
                          "edges": [{"from": edge['from'], "to": edge['to'], 'label': edge['label'], 'metadata': edge.get('title', '')} for edge in net.edges]
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


#################################################################
# Apply CSS for styling
#################################################################

    st.markdown(
        """
        <style>
            .stApp {
                max-width: 1600px;
                padding: 1rem;
            }
            .st-bf {
              margin-top: 20px;
             }

            .st-bf input, textarea {
              border: 1px solid #d3d3d3;
             }

             h1 {
                color: #333;
              }

             h3 {
                color: #444;
              }

            h4 {
                color: #555;
              }

            .css-10trblm {
                border-color:  #d3d3d3;
            }
            .css-1v3fv8k {
              border-color:  #d3d3d3;
            }

            .stButton > button {
                background-color: #5cb85c;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                cursor: pointer;
                border-radius: 0.25rem;
            }
            .stButton > button:hover {
                background-color: #4cae4c;
            }
             .stDownloadButton > button {
                 background-color: #337ab7;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                cursor: pointer;
                border-radius: 0.25rem;
             }
             .stDownloadButton > button:hover {
                background-color: #286090;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()