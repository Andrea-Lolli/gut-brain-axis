import random
import tkinter as tk

def parse_simulation_data(file_path):
    data = [] #dati della simulazione

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            agent_data = line.split()
            current_tick = int(agent_data[0])
            agent_type = int(agent_data[1])
            agent_pos = (int(agent_data[2]), int(agent_data[3]))
            #estende la lista dei dati
            if current_tick >= len(data):
                #riempe i tick vuoti (non ci dovrebbero essere boh)
                data.extend([{} for _ in range(current_tick - len(data) + 1)])
            #salva i dati per il tick corrente
            data[current_tick].setdefault(agent_type, []).append(agent_pos)

    return data


def read_network_file(file_path):
    nodes = []
    edges = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        edges_section = False
        for line in lines:
            line = line.strip()
            if line == 'EDGES':
                edges_section = True
                continue
            if not edges_section:
                if line.startswith('rumor_network'):
                    continue
                node_info = list(map(int, line.split()))
                nodes.append(node_info)
            else:
                edge_info = list(map(int, line.split()))
                edges.append(edge_info)
    return nodes, edges


class AgentModelGUI:
    def __init__(self, title, x_size, y_size, simulation_ticks, data_path, network_path):
        self.root = tk.Tk()
        self.simulation_ticks = simulation_ticks
        self.root.title(title)
        self.simulation_data = parse_simulation_data(data_path)
        self.simulation_started = False
        self.current_tick_index = -1 # usato per mostarer 1 tick per volta nella griglia
        self.x_size = x_size
        self.y_size = y_size
        self.network_path = network_path

        # Crea header
        self.header = tk.Frame(self.root)
        self.header.pack(side="top", fill="x", pady=6)
        # Bottoni header per azioni simulazione
        self.button1 = tk.Button(self.header, text="Avvia", command=self.on_start_simulation)
        self.button1.pack(side="left")
        self.button2 = tk.Button(self.header, text="Ferma", command=self.on_pause_simulation)
        self.button2.pack(side="left")
        self.button3 = tk.Button(self.header, text="Tick ++", command=self.on_advance_tick)
        self.button3.pack(side="left")
        self.button4 = tk.Button(self.header, text="Tick --", command=self.on_revert_tick)
        self.button4.pack(side="left")
        self.label1 = tk.Label(self.header, text="Tick: -")
        self.label1.pack(side="right", padx= 15)

        # Crea area per mostrare la griglia e la net
        self.container_frame = tk.Frame(self.root)
        self.container_frame.pack(side="top", fill="both", expand=True)

        # canvas per la griglia
        self.grid_canvas = tk.Canvas(self.container_frame, width=x_size * 20, height=y_size * 20, bg="white")
        self.grid_canvas.pack(side="left", fill="both", expand=True)

        # canvas per la network
        self.net_canvas = tk.Canvas(self.container_frame, width=300, height=300, bg="white")
        self.net_canvas.pack(side="left", fill="both", expand=True)

        # disegna la griglia
        self.draw_grid()

    def draw_grid(self):
        cell_width = 20 #self.grid_canvas.winfo_width() / x_size TODO
        cell_height = 20 #self.grid_canvas.winfo_height() / y_size

        for i in range(self.x_size):
            for j in range(self.y_size):
                x0 = i * cell_width
                y0 = j * cell_height
                x1 = x0 + cell_width
                y1 = y0 + cell_height
                self.grid_canvas.create_rectangle(round(x0), round(y0), round(x1), round(y1), outline="black", fill="white")

    def update_net(self):
        nodes, edges = read_network_file(self.network_path)
        node_positions = {}
        radius = 6
        levels_number = 6
        canvas_width = 300
        canvas_height = 300
        x_spacing = canvas_width // levels_number
        y_spacing = canvas_height // (len(nodes) // levels_number + 1)

        # Calculate positions for nodes
        for i, node in enumerate(nodes):
            level = i // levels_number
            x = (i % levels_number + 1) * x_spacing
            y = (level + 1) * y_spacing
            node_positions[node[0]] = (x, y)

        # Draw edges
        for edge in edges:
            node1, node2 = edge
            x1, y1 = node_positions[node1]
            x2, y2 = node_positions[node2]
            self.net_canvas.create_line(x1, y1, x2, y2)

        # Draw nodes
        for node, (x, y) in node_positions.items():
            self.net_canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="skyblue")
            self.net_canvas.create_text(x, y, text=str(node), fill="black")

    def run(self):
        self.root.mainloop()

    def draw_agents(self, agents_data):
        for agent_type, position in agents_data.items():
            agent_color = "red" if agent_type == 1 else "blue"  # TODO rendere statici da qualche parte...
            for x,y in position:
                x0 = x * 20 # TODO sarebbero da fare un po' meglio
                y0 = y * 20
                x1 = x0 + 20
                y1 = y0 + 20
                self.grid_canvas.create_rectangle(round(x0), round(y0), round(x1), round(y1), fill=agent_color)
    
    def on_start_simulation(self):
        self.simulation_started = True
        self.start_simulation()

    def start_simulation(self):
        if self.simulation_started == True:
            self.advance_tick()
            self.root.after(200, self.start_simulation)

    def on_pause_simulation(self):
        self.simulation_started = False

    def on_advance_tick(self):
        self.simulation_started = False  
        self.advance_tick()

    def advance_tick(self): 
        if self.current_tick_index < self.simulation_ticks:
            self.draw_grid()  
            self.update_net()
            self.current_tick_index += 1
            tick_data = self.simulation_data[self.current_tick_index]
            self.draw_agents(tick_data)
            self.label1.config(text=f"Tick: {self.current_tick_index}")
        else:
            self.simulation_started = False

    def on_revert_tick(self):
        self.simulation_started = False
        self.draw_grid() # Pulisci la griglia
        if self.current_tick_index >= 0:
            self.current_tick_index -= 1
            tick_data = self.simulation_data[self.current_tick_index]
            self.draw_agents(tick_data)
            self.label1.config(text=f"Tick: {self.current_tick_index}")

# TODO Da mettere come parametri letti dal file
title = "Gut-Brain Axis simulation"
x_size = 8
y_size = 8
simulation_tick = 70
app = AgentModelGUI(title, x_size, y_size, simulation_tick, "./output/test2rand.txt", "network.txt")
app.run()

# simulation_data = parse_simulation_data("exampleGuiInput.txt")
# print("dati letti in input:")
# print(simulation_data)
