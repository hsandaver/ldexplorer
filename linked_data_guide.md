# Linked Data Network Visualizer – User Guide

Welcome to the Linked Data Network Visualizer! This handy app helps you explore how people, organizations, and other items are connected. In simple terms, you can upload files that describe different items and their relationships, and the app will draw a picture (a network graph) that shows who is linked to whom.

## Getting Started

### 1. Open the App
Either run locally in a Python environment with the dependencies in requirements.txt installed using the command `streamlit run ldexplorergem2.py`, or navigate to [https://ldexplorer.streamlit.app/](https://ldexplorer.streamlit.app/).

The app uses WorldCat Entities and may require a subscription to access the full entity data from [https://id.oclc.org](https://id.oclc.org).

Since the code is open source, you can fork and modify it to work with other types of linked data.

When you launch the app, you will see a friendly homepage titled “Linked Data Network Visualizer.” This is your main workspace.

### 2. Upload Your Files
On the left sidebar (the control panel), you’ll find an option to upload JSON files.

- **What Are JSON Files?**  
JSON (JavaScript Object Notation) is a structured, human-readable format used to store and exchange data between systems. It represents data as key-value pairs, making it ideal for describing objects, arrays, and nested structures.  

In the context of this app, JSON files describe items (like people or organizations) and define how they are related to each other. There is some sample data provided. Download the sample.zip file. On Windows double click on the file then choose "extract all" to unzip the JSON files

- **How to Upload:**  
Click the “Upload JSON Files” button and select the files saved on your computer.

### 3. Visualize the Data
Once your files are loaded, the app automatically creates a visual network. Each item (or “node”) appears as a dot with its name, and lines (or “edges”) connect items that are related.

## Using the Sidebar Controls

The sidebar is where you can customize and interact with the graph. Here’s what you can do:

### 1. Filtering Relationships

- **Select Relationship Types:**  
You can choose which kinds of connections you want to see (for example, “related person” or “sibling”).

- **Filter by Entity Type:**  
If you want to focus only on people, organizations, or places, you can filter based on those categories.

### 2. Searching for Nodes

- **Search Box:**  
Enter the name of an item (for example, “Edmund Evans”) to highlight that node in the graph. This makes it easier to find specific items.

### 3. SPARQL Query (Advanced)

- **What is SPARQL?**  
SPARQL is a language used to ask questions about the data. Don’t worry—it might sound technical, but you can use simple queries to find connections.

- **How to Use:**  
There’s a text area where you can type a SPARQL query. For example, if you want to see what links two names, you can use a query that looks for any connection between them (see the examples later).

- **Note:**  
For most users, simply uploading the data and using the filters and search box will be enough. The SPARQL option is available if you’d like to dig deeper!

### 4. Manual Node Positioning

- **Adjusting Positions:**  
You can select a node and manually set its position. This is useful if you want to rearrange the graph to make it easier to understand.

- **How to Do It:**  
Pick a node from the list in the sidebar, then enter the X and Y coordinates. Click “Set Position” to update the layout.

### 5. Graph Editing

- **Adding or Removing Items and Connections:**  
The app lets you add new nodes, delete nodes, or change relationships directly in the graph.

- **Editing Options:**  
Use the “Edit Graph” section in the sidebar to choose actions like “Add Node” or “Delete Edge.”

## Exporting Your Graph

Once you have your network looking the way you want, you have options to save or share your work:

- **Download as HTML:**  
Save the graph as an HTML file that you can open in any web browser.

- **Download as JSON:**  
Export the graph data so you can share it or use it in another program.

- **Screenshot (PNG):**  
While the app doesn’t directly export images, you can always take a screenshot of the graph.

## Example SPARQL Query

If you want to ask a question about the data—for instance, “How are Edmund Evans and Richard Doyle connected?”—you can use a SPARQL query. Here’s a simplified example:

```
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ex: <http://example.org/>

SELECT DISTINCT ?connection ?connectionLabel
WHERE {
  ?s rdfs:label "Edmund Evans"@en .
  ?t rdfs:label "Richard Doyle"@en .

  ?s (ex:relatedPerson)+ ?connection .
  ?connection (ex:relatedPerson)* ?t .

  OPTIONAL { ?connection rdfs:label ?connectionLabel . }
}
```

### What This Does:

- It finds the nodes with the names “Edmund Evans” and “Richard Doyle.”
- Then, it looks for a path that connects them using the “related person” links.
- It will show you any intermediate node and its name if available.

## Troubleshooting and Tips

- **No Results?**  
If you don’t see any connections, double-check that the uploaded files have relationship information that links the items.

- **Clearing the Cache:**  
Sometimes old data can stick around. If you update your files and nothing seems to change, try clearing your browser cache or doing a hard reload.

- **Need More Help?**  
If you’re stuck, remember that the app is designed to be as simple as possible. Start with the basic upload and filtering, then explore advanced features like SPARQL queries once you’re comfortable.

## Final Thoughts

This app is all about making complex data relationships visible and easier to understand. Even if you’re not a tech expert, you can use this tool to see how different people or items are connected in a visual and interactive way.
