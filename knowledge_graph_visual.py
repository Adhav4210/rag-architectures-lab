import networkx as nx
import matplotlib.pyplot as plt

print("Step 1: Creating Knowledge Graph...")

G = nx.Graph()

print("Step 1 Completed ✅")

print("\nStep 2: Adding Entities and Relationships...")

G.add_edge("Elon Musk", "Tesla", relation="CEO")
G.add_edge("Elon Musk", "SpaceX", relation="Founder")
G.add_edge("Tesla", "Electric Vehicles", relation="Industry")
G.add_edge("SpaceX", "Rocket Technology", relation="Industry")

print("Step 2 Completed ✅")

print("\nStep 3: Visualizing Knowledge Graph...")

pos = nx.spring_layout(G)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3000,
    font_size=10
)

labels = nx.get_edge_attributes(G, "relation")

nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.title("Knowledge Graph Visualization")
plt.show()

print("Step 3 Completed ✅")

print("\nKnowledge Graph Visualization Finished 🚀")