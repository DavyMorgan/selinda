import graph_tool.all as gt

# --- Configuration ---
input_gt_files = [
    "maayan-foodweb",
    "power-eris1176",
    "advogato",
    "dblp-cite",
    "gridkit-eupowergrid",
    "gridkit-north_america",
    "ego-twitter",
    "linux",
    "internet-topology",
    "p2p-Gnutella31",
    "soc-Epinions1",
    "douban",
    "com-dblp",
    "com-amazon"
]

def convert_core(dataset):
    input_gt_file = f"{dataset}.gt"    # Path to your input .gt file
    output_txt_file = input_gt_file.replace(".gt", ".txt")  # Output .txt file path

    try:
        # 1. Load the graph from the .gt file
        print(f"Loading graph from '{input_gt_file}'...")
        g = gt.load_graph(input_gt_file)
        print(f"Graph loaded successfully: {g.num_vertices()} vertices, {g.num_edges()} edges.")

        # 2. Open the output file and write the edges
        print(f"Writing edge list to '{output_txt_file}'...")
        with open(output_txt_file, "w") as f:
            # Iterate over every edge in the graph
            for edge in g.edges():
                # Get the integer index for the source and target vertices
                source_node = int(edge.source())
                target_node = int(edge.target())

                # Write the formatted line to the file
                # Note: {{}} is used in f-strings to produce a literal {}
                if source_node != target_node:  # Avoid self-loops
                    f.write(f"{source_node} {target_node} {{}}\n")

        print("Conversion complete.")

    except FileNotFoundError:
        print(f"Error: The file '{input_gt_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

for dataset in input_gt_files:
    convert_core(dataset)

