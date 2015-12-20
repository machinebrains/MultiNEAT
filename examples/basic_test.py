
import MultiNEAT as neat
params = neat.Parameters()

params.PopulationSize = 100

genome = neat.Genome(0,3,0,2,False,neat.ActivationFunction.UNSIGNED_SIGMOID,neat.ActivationFunction.UNSIGNED_SIGMOID,0,params)
pop = neat.Population(genome,params, True, 1.0, 1)

def evaluate(genome):
    net = neat.NeuralNetwork()
    genome.BuildPhenotype(net)
    net.Input([1.0,0.0,1.0])
    net.Activate()
    output = net.Output()
    
    fitness = 1.0 - output[0]
    return fitness

for generation in range(100):
    genome_list = neat.GetGenomeList(pop)
    
    for genome in genome_list:
        fitness = evaluate(genome)
        genome.SetFitness(fitness)
        print fitness
    
    pop.Epoch()


