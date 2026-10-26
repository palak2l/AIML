import networkx as nx
import matplotlib.pyplot as plt

facts = {'fever', 'cough', 'fatigue'}

rules = {
    'Flu': {'fever', 'cough'},
    'Infection': {'fever', 'fatigue'}
}

def forward():
    return [d for d, s in rules.items() if s.issubset(facts)]

def backward(goal):
    return rules[goal].issubset(facts)

print("Forward Chaining:", forward())
print("Backward Chaining (Flu):", backward('Flu'))

G = nx.DiGraph()
for disease, symptoms in rules.items():
    for symptom in symptoms:
        G.add_edge(symptom, disease)

pos = nx.spring_layout(G, seed=42)

nx.draw(G, pos, with_labels=True,
        node_color="lightgreen", node_size=2000,
        arrows=True, font_size=9)

plt.title("Expert System - Rule Graph")
plt.show()