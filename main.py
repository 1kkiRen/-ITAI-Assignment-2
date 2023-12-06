import random
import os

# Global variables  
POPULATION_SIZE = 250
GENERATIONS = 15000
SIZE = 20
ELITISM = 0.1
MUTATION_RATE = 0.15
TOURNAMENT_SIZE = 5
words = []

class Crossword:
    """ Class that represents the crossword. """
    def __init__(self):
        """ Constructor. """
        self.grid = []
        self.fitness = -10 ** 9
    
    def init_the_crossword(self):
        """ Initialize the crossword. """
        for _ in range(SIZE):
            self.grid.append(['-'] * SIZE)
            
    def append(self, row):
        """ Append a row to the crossword.

        Args:
            row (list): The row to be appended.
        """
        self.grid.append(row)
        
    def print(self):
        """ Print the crossword. """
        for row in self.grid:
            print(''.join(row))
        
    def copy_to(self, crossword):
        """ Copy the crossword to another crossword.

        Args:
            crossword (Crossword): The crossword to be copied to.
        """
        for i in range(SIZE):
            for j in range(SIZE):
                crossword.grid[i][j] = self.grid[i][j]
                    
            crossword.fitness = self.fitness
            

class Individual:
    """ Class that represents the individual. """
    def __init__(self):
        """ Constructor. """
        self.content = []
        self.crossword = None
        self.warning_words = []
        
    def init_the_individual(self):
        """ Initialize the individual. """
        self.crossword = Crossword()
        self.crossword.init_the_crossword()
        for word in words:
            while True:
                row = random.randint(0, SIZE - 1)
                col = random.randint(0, SIZE - 1)
                direction = random.randint(0, 1)
                
                if self.crossword.grid[row][col] not in ['-', word[0]]:
                    continue
                
                if direction == 0:
                    if col + len(word) > SIZE:
                        continue
                
                    for i in range(len(word)):
                        self.crossword.grid[row][col + i] = word[i]
                        
                else:
                    if row + len(word) > SIZE:
                        continue
                    
                    for i in range(len(word)):
                        self.crossword.grid[row + i][col] = word[i]
                
                break
            
            self.content.append({'word': word, 'row': row, 'col': col, 'direction': direction})
    
        
    def print(self):
        """ Print the individual. """
        self.crossword.print()

def read_words(i):
    """ Read the words from the input file.

    Args:
        i (int): The number of the input file.

    Returns:
        list: The list of the words.
    """
    words = []
    with open(f'inputs/input{i + 1}.txt', 'r') as f:
        for line in f:
            words.append(line.strip())
    return words

def fitness(individual, debug):
    """ Calculate the fitness of the individual.

    Args:
        individual (Individual): The individual to calculate the fitness of.
        debug (boolean): Whether to print the debug info or not.
    """
    fitness = 0

    horizontal_substrings = []
    vertical_substrings = []
    for i in range(SIZE):
        horizontal_substring = ''
        for j in range(SIZE):
            if individual.crossword.grid[i][j] == '-':
                continue
            else:
                horizontal_substring += individual.crossword.grid[i][j]
                if j == SIZE - 1 or individual.crossword.grid[i][j + 1] == '-':
                    horizontal_substrings.append(horizontal_substring) if len(horizontal_substring) > 1 else None
                    horizontal_substring = ''
                    
    for j in range(SIZE):
        vertical_substring = ''
        for i in range(SIZE):
            if individual.crossword.grid[i][j] == '-':
                continue
            else:
                vertical_substring += individual.crossword.grid[i][j]
                if i == SIZE - 1 or individual.crossword.grid[i + 1][j] == '-':
                    vertical_substrings.append(vertical_substring) if len(vertical_substring) > 1 else None
                    vertical_substring = ''
    
    for substring in [*horizontal_substrings, *vertical_substrings]:
        if substring in words:
            fitness += 2000
        else:
            fitness -= 1000000
            
    for word in words:
        if word not in [*horizontal_substrings, *vertical_substrings]:
            fitness -= 1000000

    all_words_flag = False if fitness < len(words) * 2000 else True

    if not all_words_flag:
        fitness -= 1000000
        
    collisions = {word: [] for word in words}
    
    count_of_intersections = 0
    for i in range(len(individual.content)):
        for j in range(len(individual.content)):
            if i == j:
                continue
            
            if individual.content[i]['direction'] == individual.content[j]['direction']:
                continue
            
            if individual.content[i]['direction'] == 0:
                if individual.content[i]['row'] >= individual.content[j]['row'] and individual.content[i]['row'] <= individual.content[j]['row'] + len(individual.content[j]['word']) - 1:
                    if individual.content[j]['col'] >= individual.content[i]['col'] and individual.content[j]['col'] <= individual.content[i]['col'] + len(individual.content[i]['word']) - 1:
                        collisions[individual.content[i]['word']].append(individual.content[j]['word'])
                        if debug:
                            print(individual.content[i]['word'], individual.content[j]['word'])
            else:
                if individual.content[i]['col'] >= individual.content[j]['col'] and individual.content[i]['col'] <= individual.content[j]['col'] + len(individual.content[j]['word']) - 1:
                    if individual.content[j]['row'] >= individual.content[i]['row'] and individual.content[j]['row'] <= individual.content[i]['row'] + len(individual.content[i]['word']) - 1:
                        collisions[individual.content[i]['word']].append(individual.content[j]['word'])
                        if debug:
                            print(individual.content[i]['word'], individual.content[j]['word'])
                        
    for word in collisions:
        if len(collisions[word]) > 0:
            count_of_intersections += 1
            fitness += 100 * len(collisions[word])
        else:
            fitness -= 10000
        
    isolated_words = [word for word, intersections in collisions.items() if len(intersections) == 0]

    if debug:
        print(isolated_words)
    
    fitness -= len(isolated_words) * 40000
    
    def DFS(word, visited) -> None:
        """ DFS algorithm.

        Args:
            word (string): The word to start the DFS from.
            visited (dict): The dictionary of the visited words.
        """
        visited[word] = True
        for neighbour in collisions[word]:
            if not visited[neighbour]:
                DFS(neighbour, visited)
                
    visited = {word: False for word in words}
    DFS(words[0], visited)
    
    for word in visited:
        if visited[word]:
            fitness += 1000
        if not visited[word]:
            fitness -= 100000
    
    isolated_subcrosswords = []
    
    for word in words:
        if word in isolated_words:
            continue
        visited = {word: False for word in words}
        DFS(word, visited)

        if not all(visited.values()):
            isolated_subcrosswords.append(visited)
    
    isolated_subcrosswords = [dict(t) for t in {tuple(d.items()) for d in isolated_subcrosswords}]
    
    fitness -= len(isolated_subcrosswords) * 40000
    
    biggest_subcrossword = None
    
    for subcrossword in isolated_subcrosswords:
        count = sum(subcrossword.values())
        if biggest_subcrossword is None or count > sum(biggest_subcrossword.values()):
            biggest_subcrossword = subcrossword
    
    warning_words = []
    
    if biggest_subcrossword is not None:
        for word in biggest_subcrossword:
            if not biggest_subcrossword[word]:
                warning_words.append(word)


    if all_words_flag and False not in visited.values():
        fitness += 1000000
        
    if debug:
        print()
        print("Debug info:")
        print(collisions)
        print(visited)
        print(all_words_flag)
        print()
        print(fitness)
        print()
    
    if debug:
        print(count_of_intersections)
        
    individual.crossword.fitness = fitness
    individual.warning_words = warning_words.copy()
            
def generate_initial_population() -> list:
    """ Generate the initial population.

    Returns:
        list: The initial population.
    """
    population = []
    
    for _ in range(POPULATION_SIZE):
        individual = Individual()
        individual.init_the_individual()
        population.append(individual)
         
    return population

def selection(population):
    """ Select the elites.

    Args:
        population (list[Individual]): The population to select the elites from.

    Returns:
        list[Individual]: The elites.
    """
    population.sort(key=lambda x: x.crossword.fitness, reverse=True)
    elites = population[:int(POPULATION_SIZE * ELITISM) + 1]
    return elites

def select_parents(elites):
    """ Select the parents.

    Args:
        elites (list[Individual]): The elites to select the parents from.

    Returns:
        list[Individual]: The parents.
    """
    parents = []
    for _ in range(2):
        while True:
            parent = tournament_selection(elites)
            if parent not in parents:
                parents.append(parent)
                break

    return parents

def crossover(population, elites):
    """ Crossover the population.

    Args:
        population (list[Individual]): The population to crossover.
        elites (list[Individual]): The elites to crossover.

    Returns:
        list[Individual]: The new population.
    """
    new_population = []
    
    for _ in range(POPULATION_SIZE):
        parents = select_parents(elites)
        
        individual = Individual()
        individual.crossword = Crossword()
        individual.crossword.init_the_crossword()
        
        for i in range(len(parents[0].content)):
            individual.content.append({'word': population[_].content[i]['word'], 'row': 0, 'col': 0, 'direction': 0})

        for i in range(len(parents[0].content)):
            if random.random() < 0.5:
                individual.content[i]['row'] = parents[0].content[i]['row']
            else:
                individual.content[i]['row'] = parents[1].content[i]['row']
        
        for i in range(len(parents[0].content)):
            if random.random() < 0.5:
                individual.content[i]['col'] = parents[0].content[i]['col']
            else:
                individual.content[i]['col'] = parents[1].content[i]['col']
                
        for i in range(len(parents[0].content)):
            if random.random() < 0.5:
                individual.content[i]['direction'] = parents[0].content[i]['direction']
            else:
                individual.content[i]['direction'] = parents[1].content[i]['direction']
                
        for i in range(len(parents[0].content)):
            word = individual.content[i]['word']
            row = individual.content[i]['row']
            col = individual.content[i]['col']
            direction = individual.content[i]['direction']
            
            if direction == 0:
                for j in range(len(word)):
                    if col + len(word) >= SIZE:
                        delta = col + len(word) - SIZE + 1
                        col -= delta
                        individual.content[i]['col'] = col
                        
                    individual.crossword.grid[row][col + j] = word[j]
            else:
                for j in range(len(word)):
                    if row + len(word) >= SIZE:
                        delta = row + len(word) - SIZE + 1
                        row -= delta
                        individual.content[i]['row'] = row
                        
                    individual.crossword.grid[row + j][col] = word[j]
                    
        new_population.append(individual)
                    
    return new_population

def mutation(population, mutation_rate):
    """ Mutate the population.

    Args:
        population (list[Individual]): The population to mutate.
        mutation_rate (float): The mutation rate.

    Returns:
        list[Individual]: The mutated population.
    """
    for individual in population:
        for i in range(len(individual.content)):
            if random.random() < mutation_rate or individual.content[i]['word'] in individual.warning_words:
                individual.content[i]['row'] = random.randint(0, SIZE - 1)
                individual.content[i]['col'] = random.randint(0, SIZE - 1)
                individual.content[i]['direction'] = random.randint(0, 1)
                
                new_crossword = Crossword()
                new_crossword.init_the_crossword()

                for j in range(len(individual.content)):
                    word = individual.content[j]['word']
                    row = individual.content[j]['row']
                    col = individual.content[j]['col']
                    direction = individual.content[j]['direction']
                    
                    if direction == 0:
                        for k in range(len(word)):
                            if col + len(word) >= SIZE:
                                delta = col + len(word) - SIZE + 1
                                col -= delta
                                individual.content[j]['col'] = col
                                
                            new_crossword.grid[row][col + k] = word[k]
                    else:
                        for k in range(len(word)):
                            if row + len(word) >= SIZE:
                                delta = row + len(word) - SIZE + 1
                                row -= delta
                                individual.content[j]['row'] = row
                                
                            new_crossword.grid[row + k][col] = word[k]
                            
                new_crossword.copy_to(individual.crossword)
                
    return population

def tournament_selection(population):
    """ Select the individual from the tournament.

    Args:
        population (list[Individual]): The population to select the individual from.

    Returns:
        Individual: The selected individual.
    """
    tournament = []
    for _ in range(TOURNAMENT_SIZE):
        while True:
            individual = random.choice(population)
            if individual not in tournament:
                tournament.append(individual)
                break
    
    tournament.sort(key=lambda x: x.crossword.fitness, reverse=True)
    return tournament[0]

def sort_population(population):
    """ Sort the population.

    Args:
        population (list[Individual]): The population to sort.

    Returns:
        list[Individual]: The sorted population.
    """
    population.sort(key=lambda x: x.crossword.fitness, reverse=True)
    return population

def run_test(i):
    """ Run the test.

    Args:
        i (int): The number of the input file.

    Returns:
        int: The fitness of the best individual.
    """
    global words

    words = read_words(i)    

    fitnesses = []
    population = generate_initial_population()    
    best_individual = Individual()
    best_individual.crossword = Crossword()
    best_individual.crossword.init_the_crossword()

    crossword_repetitions = 0
    fitness_repetitions = 0

    previous_individual = Individual()
    previous_individual.crossword = Crossword()
    previous_individual.crossword.init_the_crossword()

    for individual in population:
        fitness(individual, False)

    for generation in range(GENERATIONS):
        MUTATION_RATE = 0.15
                
        elites = selection(population)
    
        population = crossover(population, elites)
            
        if crossword_repetitions > 200 or fitness_repetitions > 250:
            MUTATION_RATE = 0.65
            crossword_repetitions = 0
            
        population = mutation(population, MUTATION_RATE)   
            
        for individual in population:
            fitness(individual, False)
            
        sort_population(population)
        
        if previous_individual.crossword.grid == population[0].crossword.grid:
            crossword_repetitions += 1
        else:
            crossword_repetitions = 0
        
        if previous_individual.crossword.fitness == population[0].crossword.fitness:
            fitness_repetitions += 1
        else:
            fitness_repetitions = 0
        
        fitnesses.append(population[0].crossword.fitness)

        if population[0].crossword.fitness > best_individual.crossword.fitness:
            best_individual = population[0]
            
        if population[0].crossword.fitness > 1000000:
            break
        
        previous_individual = population[0]

    
    with open(f'outputs/output{i + 1}.txt', "w") as output_file:
        for word in words:
            output_file.write(f'{best_individual.content[words.index(word)]["row"]} {best_individual.content[words.index(word)]["col"]} {best_individual.content[words.index(word)]["direction"]}\n')
    
    with open(f'output_crosswords/output{i + 1}.txt', "w") as output_file:
        for row in best_individual.crossword.grid:
            output_file.write(f'{" ".join(row)}\n')
    
    print(f'Output{i + 1} is written')
    
    return best_individual.crossword.fitness    


if __name__ == '__main__':
    """ Main function. """
    total_fitness = []
    
    input_files = len(os.listdir('inputs'))
    
    print(f'Number of input files: {input_files}')  
    
    for i in range(input_files):
        print(f'Input{i + 1} is read')
        total_fitness.append(run_test(i))
        
    print(f'Total fitness: {sum(total_fitness)}')
    print(f'Average fitness: {sum(total_fitness) / input_files}')
    print(f'Max fitness: {max(total_fitness)}')
    print(f'Min fitness: {min(total_fitness)}')