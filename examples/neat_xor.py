##################################################################
# Testing NEAT with XOR
##################################################################
import os
import sys
import time
import random as rnd
import cv2
import numpy as np
import pickle as pickle
import MultiNEAT as NEAT
from MultiNEAT import EvaluateGenomeList_Serial
from MultiNEAT import GetGenomeList, ZipFitness

from concurrent.futures import ProcessPoolExecutor, as_completed


def evalGenome(genome, inputs):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)
    
    outputs = []
    for inp in inputs:
        net.Flush()
        net.Input(np.array(inp))
        for _ in range(len(inputs)):
            net.Activate()
        o = net.Output()
        outputs.append(o[0])
    return outputs

def absErr(y,yhat):
    diffs = np.abs(np.array(y) - np.array(yhat))
    error = np.sum(diffs)
    return (4.0 - error)**2.0
    
def evaluate(genome,inputs,outputs):
    yhat = evalGenome(genome, inputs)
    return absErr(outputs,yhat)
    
    

params = NEAT.Parameters()
params.PopulationSize = 150
params.DynamicCompatibility = True
params.WeightDiffCoeff = 4.0
params.CompatTreshold = 2.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 15
params.OldAgeTreshold = 35
params.MinSpecies = 5
params.MaxSpecies = 10
params.RouletteWheelSelection = False
params.RecurrentProb = 0.0
params.OverallMutationRate = 0.8

params.MutateWeightsProb = 0.90

params.WeightMutationMaxPower = 2.5
params.WeightReplacementMaxPower = 5.0
params.MutateWeightsSevereProb = 0.5
params.WeightMutationRate = 0.25

params.MaxWeight = 8

params.MutateAddNeuronProb = 0.03
params.MutateAddLinkProb = 0.05
params.MutateRemLinkProb = 0.0

params.MinActivationA  = 4.9
params.MaxActivationA  = 4.9

params.ActivationFunction_SignedSigmoid_Prob = 0.0
params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
params.ActivationFunction_Tanh_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 0.0

params.CrossoverRate = 0.75  # mutate only 0.25
params.MultipointCrossoverRate = 0.4
params.SurvivalRate = 0.2


def getbest(i):
    g = NEAT.Genome(0, 2, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)
    pop = NEAT.Population(g, params, True, 1.0, i)
    pop.RNG.Seed(rnd.randint(1,5000))

    generations = 0
    fitness_list = []
    Reached_Best = 0
    bestInd = None
    inputs = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
    outputs = [0.0,1.0,1.0,0.0]
    
    for generation in range(1000):
        genome_list = NEAT.GetGenomeList(pop)
        fitness_list = EvaluateGenomeList_Serial(genome_list, evaluate,inputs,outputs, display=False)
        
        best = max(fitness_list)
        
        #print best

        if best > 15.0:
            print "Best Found = ", best
            bestI = fitness_list.index(best)
            print "at = ", bestI, " with value = " , fitness_list[bestI]
            bestInd = genome_list[bestI]
            
            print "# Evaluating best value."
            bestValues = evalGenome(bestInd,inputs)
            error = absErr(outputs,bestValues)
            print "Inputs = " , inputs
            print "Best Output = ", bestValues
            print "Best Error = ", error
            
            
            Reached_Best = 1
            break
        
        NEAT.ZipFitness(genome_list, fitness_list)
        pop.Epoch()
        generations = generation
    
    print Reached_Best
    return generations



gens = []
for run in range(5):
    gen = getbest(run)
    gens += [gen]
    print('Run:', run, 'Generations to solve XOR:', gen)
avg_gens = sum(gens) / len(gens)

print('All:', gens)
print('Average:', avg_gens)


