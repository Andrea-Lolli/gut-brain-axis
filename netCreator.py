import random

# Costante che serve per gli edges
ranges = [
    (0, 199),
    (200, 399),
    (400, 599),
    (600, 799)
]

def generate_neural_net_file(name, neurons_number, ranks_number):
    file_name = f"{name}.txt"
    with open(file_name, 'w') as file:
        file.write(f"{name} 0\n")
        
        x = neurons_number*2
        # Calcola il numero di neuroni per ogni rank
        if ranks_number >= neurons_number:
            for i in range(x):
                value = 3 if i % 2 == 0 else 4
                file.write(f"{i} {value} {i}\n")
        else:
            quotient = x // ranks_number
            remainder = x % ranks_number
            current_group = 0
            for i in range(x):
                value = 3 if i % 2 == 0 else 4
                file.write(f"{i} {value} {current_group}\n")
                if (i + 1) % quotient == 0 and remainder > 0:
                    remainder -= 1
                    current_group += 1
                elif (i + 1) % quotient == 0:
                    current_group += 1
        
        # Aggiunge tag EDGES
        file.write("EDGES\n")
        
        # Collega glia a neuroni
        for i in range(0, neurons_number * 2, 2):
            if i + 1 < neurons_number * 2:
                file.write(f"{i} {i + 1}\n")

        # TODO non sono parametrizzati! funziona solo per 4 ranks...
        # (in caso c'è da aggiustare la variabile ranges)
        for start, end in ranges:
            for _ in range(200):
                x = random.randint(start // 2, end // 2) * 2
                y = random.randint(start // 2, end // 2) * 2
                file.write(f"{x} {y}\n")
        
generate_neural_net_file("testNet", 400, 4) # Genera 400 neuroni + 400 glia in 4 rank 