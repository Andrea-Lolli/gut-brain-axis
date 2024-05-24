import random
import argparse
import tkinter as tk

def parse_simulation_data(file_path, ranks):
    gut_data = [] # Dati intestino (batteri benefici e nocivi)
    soglia_data = {} # Valori della simulazione

    for rank in ranks:
        soglia_rank = []
        with open(file_path, 'r') as file:

            for line in file:
                line = line.strip()
                agent_data = line.split()
                current_tick = int(agent_data[0])
                agent_type = int(agent_data[1])

                if agent_type == 1 or agent_type == 2: # Agente = batteri
                    agent_pos = (int(agent_data[2]), int(agent_data[3]))
                    # Estende la lista dei dati
                    if current_tick >= len(gut_data):
                        # Riempe i tick vuoti (non ci dovrebbero essere boh)
                        gut_data.extend([{} for _ in range(current_tick - len(gut_data) + 1)])
                    # Salva i dati per il tick corrente
                    gut_data[current_tick].setdefault(agent_type, []).append(agent_pos)

                elif agent_type == 0 and int(agent_data[8]) == rank: # Dati di un agente Soglie
                    nutrienti_b = agent_data[2]
                    nutrienti_n = agent_data[3]
                    prodotto_b = agent_data[4]
                    prodotto_n = agent_data[5]
                    citochine_g = agent_data[6]
                    citochine_b = agent_data[7]
                    soglia_rank.append((current_tick, nutrienti_b, nutrienti_n, prodotto_b, prodotto_n, citochine_g, citochine_b))
        soglia_data[rank] = soglia_rank

    return gut_data, soglia_data


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
    def __init__(self, title, x_size, y_size, simulation_ticks, data_path, network_path, gut_view, network_view, chart_view, ranks):
        self.root = tk.Tk()
        self.simulation_ticks = simulation_ticks
        self.root.title(title)
        self.simulation_data, self.soglia_data = parse_simulation_data(data_path, ranks)
        self.simulation_started = False
        self.current_tick_index = -1 # usato per mostarer 1 tick per volta nella griglia
        self.x_size = x_size
        self.y_size = y_size
        self.network_path = network_path
        self.gut_view = gut_view
        self.network_view = network_view
        self.chart_view = chart_view
        self.ranks = ranks

        #TODO migliorare che fa schifo
        lb, ln, cg, cb,  cito_g, cito_b = [], [], [], [], [], []
        nRanks = len(self.ranks)
        n_items = len(list(self.soglia_data.values())[0])
        
        for i in range(n_items):
            lb0, ln0, cg0, cb0, cito_g0, cito_b0 = 0,0,0,0,0,0
            for rank in self.ranks:
                data = self.soglia_data[rank]
                lb0 += float(data[i][1])
                ln0 += float(data[i][2])
                cg0 += float(data[i][3])
                cb0 += float(data[i][4])
                cito_g0 += float(data[i][5])
                cito_b0 += float(data[i][6])
            lb.append(lb0/nRanks)
            ln.append(ln0/nRanks)
            cg.append(cg0/nRanks)
            cb.append(cb0/nRanks)
            cito_g.append(cito_g0/nRanks)
            cito_b.append(cito_b0/nRanks)

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
        
        # Gut
        if gut_view and x_size < 50 and y_size < 50: #se la griglia ha dimensioni ridotte
            self.grid_canvas = tk.Canvas(self.container_frame, width=40+x_size * 20, height=40+y_size * 20, bg="white")
            self.grid_canvas.pack(side="left", fill="both", expand=True)
            self.draw_grid()

        # Network
        if network_view:
            self.net_canvas = tk.Canvas(self.container_frame, width=300, height=300, bg="white")
            self.net_canvas.pack(side="left", fill="both", expand=True)
            self.draw_net()

        # Charts
        if chart_view:
            self.chart_frame = tk.Frame(self.root)
            self.chart_frame.pack(side="bottom", fill="both", expand=True)
            self.chart_canvas = tk.Canvas(self.chart_frame, width=600, height=200, bg="white")
            self.chart_canvas.pack(side="top", fill="both", expand=True)
            #self.plot_line_chart((lb, ln), ("blue", "red")) #TODO test
            #self.draw_chart() # ???
            self.plot_line_chart((cg, cb), ("orange", "yellow"))  
            self.plot_line_chart((cito_g, cito_b), ("red", "green"))

    def draw_grid(self):
        cell_width = 20 #self.grid_canvas.winfo_width() / x_size TODO
        cell_height = 20 #self.grid_canvas.winfo_height() / y_size
        padding = 20

        for i in range(self.x_size):
            for j in range(self.y_size):
                x0 = i * cell_width
                y0 = j * cell_height
                x1 = x0 + cell_width
                y1 = y0 + cell_height
                self.grid_canvas.create_rectangle(padding + round(x0), padding + round(y0), padding + round(x1), padding + round(y1), outline="black", fill="white")

    def draw_net(self):
        nodes, edges = read_network_file(self.network_path)
        if nodes > 100: # evitiamo di disegnare un milione di nodi...
            return
        node_positions = {}
        radius = 6
        levels_number = 6
        canvas_width = 300
        canvas_height = 300
        x_spacing = canvas_width // levels_number
        y_spacing = canvas_height // (len(nodes) // levels_number + 1)

        # Calcola la posizione dei nodi
        for i, node in enumerate(nodes):
            level = i // levels_number
            x = (i % levels_number + 1) * x_spacing
            y = (level + 1) * y_spacing
            node_positions[node[0]] = (x, y)

        # Disegna vertici
        for edge in edges:
            node1, node2 = edge
            x1, y1 = node_positions[node1]
            x2, y2 = node_positions[node2]
            self.net_canvas.create_line(x1, y1, x2, y2)

        # Disegna nodi
        for node, (x, y) in node_positions.items():
            self.net_canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="skyblue")
            self.net_canvas.create_text(x, y, text=str(node), fill="black")

    def plot_line_chart(self, data_sets, colors):
        #self.chart_canvas.delete("all")  # Clear the previous lines
        if not data_sets or not colors:
            return

        # Determine canvas size and padding
        width = int(self.chart_canvas['width'])
        height = int(self.chart_canvas['height'])
        padding = 30
        plot_width = width - 2 * padding
        plot_height = height - 2 * padding

        # Determine global min and max values for scaling
        all_values = [value for data_points in data_sets for value in data_points]
        max_value = max(all_values)
        min_value = min(all_values)
        value_range = max_value - min_value if max_value != min_value else 1

        x_steps = [plot_width / (len(data_points) - 1) for data_points in data_sets]
        y_scale = plot_height / value_range

        # Draw x and y axes
        self.chart_canvas.create_line(padding, height - padding, width - padding, height - padding, fill="black")  # x-axis
        self.chart_canvas.create_line(padding, padding, padding, height - padding, fill="black")  # y-axis

        # Add labels to axes
        #self.chart_canvas.create_text(padding, height - padding, text=str(min_value), anchor=tk.E)
        #self.chart_canvas.create_text(padding, padding, text=str(max_value), anchor=tk.E)

        max_len = max(len(data_points) for data_points in data_sets)
        #self.chart_canvas.create_text(width - padding, height - padding / 2, text=str(max_len - 1), anchor=tk.N)

        # Plot each dataset
        for data_points, color in zip(data_sets, colors):
            x_step = plot_width / (len(data_points) - 1)
            for i in range(len(data_points) - 1):
                x1 = padding + i * x_step
                y1 = height - padding - (data_points[i] - min_value) * y_scale
                x2 = padding + (i + 1) * x_step
                y2 = height - padding - (data_points[i + 1] - min_value) * y_scale
                self.chart_canvas.create_line(x1, y1, x2, y2, fill=color)

    def draw_agents(self, agents_data):
        padding = 20 # TODO dovrebbe essere uguale all'altro

        for agent_type, position in agents_data.items():
            agent_color = "red" if agent_type == 1 else "blue"  # TODO rendere statici da qualche parte...
            for x,y in position:
                x0 = x * 20 # TODO sarebbero da fare un po' meglio
                y0 = y * 20
                x1 = x0 + 20
                y1 = y0 + 20
                self.grid_canvas.create_rectangle(padding + round(x0), padding + round(y0), padding + round(x1), padding + round(y1), fill=agent_color)
    
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
            self.current_tick_index += 1
            tick_data = self.simulation_data[self.current_tick_index]
            self.label1.config(text=f"Tick: {self.current_tick_index}")
            if self.gut_view: # aggiorna griglia
                self.draw_grid()  
                self.draw_agents(tick_data)
        else:
            self.simulation_started = False

    def on_revert_tick(self):
        self.simulation_started = False
        if self.current_tick_index >= 0:
            self.current_tick_index -= 1
            tick_data = self.simulation_data[self.current_tick_index]
            self.label1.config(text=f"Tick: {self.current_tick_index}")
            if self.gut_view: # aggiorna griglia
                self.draw_grid() 
                self.draw_agents(tick_data)
    
    def run(self):
        self.root.mainloop()

# TODO I default andrebbero rivisti ma vabè

def main():
    parser = argparse.ArgumentParser(description='Run the Gut-Brain Axis simulation.')
    parser.add_argument('-t','--title', type=str, default='Gut-Brain Axis simulation', help='Title of the simulation window')
    parser.add_argument('--x_size', type=int, default=8, help='X size of the grid')
    parser.add_argument('--y_size', type=int, default=8, help='Y size of the grid')
    parser.add_argument('--simulation_ticks', type=int, default=100, help='Number of simulation ticks')
    parser.add_argument('-i', '--input_file', type=str, required=True, help='Path to the input file')
    parser.add_argument('-nf', '--network_file', type=str, help='Path to the network file')
    parser.add_argument('-g','--gut_view', action='store_true', required=False, help='Enable gutgui')
    parser.add_argument('-n', '--network_view', action='store_true', required=False, help='Enable network gui')
    parser.add_argument('-c', '--chart_view', action='store_true', required=False, help='Enable chart')
    parser.add_argument('-r','--rank', type=str, default='', required=False, help='Select ranks to show in chart (ex: 1,2,3,4)')
    
    args = parser.parse_args()
    
    rnk_str = args.rank.split(',') #rimuovi valori non nuerici dai rank
    ranks = []
    for elemento in rnk_str:
        try:
            numero = int(elemento)
            ranks.append(numero)
        except ValueError:
            pass

    app = AgentModelGUI(
        title=args.title,
        x_size=args.x_size,
        y_size=args.y_size,
        simulation_ticks=args.simulation_ticks,
        data_path=args.input_file,
        network_path=args.network_file,
        gut_view=args.gut_view,
        network_view=args.network_view,
        chart_view=args.chart_view,
        ranks = ranks
    )
    
    app.run()

if __name__ == "__main__":
    main()
