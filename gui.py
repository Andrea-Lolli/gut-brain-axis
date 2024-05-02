import tkinter as tk

def parse_simulation_data(file_path):
    data = []  # Array to store data for each tick
    current_tick_data = {}  # Dictionary to store data for the current tick
    current_tick = None

    with open(file_path, 'r') as file:
        isData = False #avoid reading general simulation info
        for line in file:
            line = line.strip()
            if line.startswith("TICK"):
                isData = True
                if current_tick is not None:
                    data.append(current_tick_data)
                current_tick = int(line.split()[1])
                current_tick_data = {}
            elif line and isData:
                agent_data = line.split()
                agent_type = int(agent_data[0])
                agent_pos = (int(agent_data[1]), int(agent_data[2]))
                current_tick_data.setdefault(agent_type, []).append(agent_pos)

        if current_tick is not None:
            data.append(current_tick_data)

    return data


class AgentModelGUI:
    def __init__(self, title, x_size, y_size, data_path):
        self.root = tk.Tk()
        self.root.title(title)
        self.simulation_data = parse_simulation_data(data_path)
        self.current_tick_index = -1 # usato per mostarer 1 tick per volta nella griglia

        # Crea header
        self.header = tk.Frame(self.root)
        self.header.pack(side="top", fill="x", pady=6)
        # Bottoni header per azioni simulazione
        self.button1 = tk.Button(self.header, text="Avvia")
        self.button1.pack(side="left")
        self.button2 = tk.Button(self.header, text="Ferma")
        self.button2.pack(side="left")
        self.button3 = tk.Button(self.header, text="Tick ++", command=self.advance_tick)
        self.button3.pack(side="left")
        self.button4 = tk.Button(self.header, text="Tick --")
        self.button4.pack(side="left")
        self.label1 = tk.Label(self.header, text="Tick: -")
        self.label1.pack(side="right", padx= 15)

        # Crea area per mostrare la griglia
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(side="top", fill="both", expand=True)

        self.grid_canvas = tk.Canvas(self.grid_frame, width=x_size * 20, height=y_size * 20)  # TODO forse il moltiplicatore è da cambiare
        self.grid_canvas.pack(fill="both", expand=True)
        #self.grid_canvas.create_rectangle(10, 10, 30, 30, outline="black", fill="blue")

        # disegna la griglia
        self.draw_grid(x_size, y_size)

    def draw_grid(self, x_size, y_size):
        cell_width = 10 #self.grid_canvas.winfo_width() / x_size TODO
        cell_height = 10 #self.grid_canvas.winfo_height() / y_size

        for i in range(x_size):
            for j in range(y_size):
                x0 = i * cell_width
                y0 = j * cell_height
                x1 = x0 + cell_width
                y1 = y0 + cell_height
                self.grid_canvas.create_rectangle(round(x0), round(y0), round(x1), round(y1), outline="black")

    def run(self):
        self.root.mainloop()

    def draw_agents(self, agents_data):
        for agent_type, position in agents_data.items():
            agent_color = "red" if agent_type == 1 else "blue"  # TODO rendere statici da qualche parte...
            for x,y in position:
                x0 = x * 10 # TODO sarebbero da fare un po' meglio
                y0 = y * 10
                x1 = x0 + 10
                y1 = y0 + 10
                self.grid_canvas.create_rectangle(round(x0), round(y0), round(x1), round(y1), fill=agent_color)
    
    def start_simulation(self):
        #TODO da implementare
        print("Avvio la simulazione...")

    def pause_simulation(self):
        #TODO da implementare
        print("fermo la simulazione...")

    def advance_tick(self):
        self.current_tick_index += 1

        if self.current_tick_index < len(self.simulation_data):
            tick_data = self.simulation_data[self.current_tick_index]
            self.draw_agents(tick_data)
            self.label1.config(text=f"Tick: {self.current_tick_index}")

    def revert_tick(self):
        #TODO da implementare
        print("Reverting tick...")

# TODO Da mettere come parametri letti dal file
title = "Gut-Brain Axis simulation"
x_size = 20
y_size = 20
app = AgentModelGUI(title, x_size, y_size, "exampleGuiInput.txt")
app.run()

# simulation_data = parse_simulation_data("exampleGuiInput.txt")
# print("dati letti in input:")
# print(simulation_data)
