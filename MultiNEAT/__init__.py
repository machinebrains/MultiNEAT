import time
from _MultiNEAT import *
import multiprocessing as mp


# Get all genomes from the population
def GetGenomeList(pop):
    genome_list = []
    for s in pop.Species:
        for i in s.Individuals:
            genome_list.append(i)
    return genome_list
    
# Just set the fitness values to the genomes
def ZipFitness(genome_list, fitness_list):
    [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitness_list)]
    [genome.SetEvaluated() for genome in genome_list]

RetrieveGenomeList = GetGenomeList
FetchGenomeList = GetGenomeList

try:
    from IPython.display import clear_output
    from ipyparallel import Client
    ipython_installed = True
except:
    ipython_installed = False


# Evaluates fitness for a single individual.
def EvaluateGenomeFitness(genome,evaluator,inputs=None, outputs=None):
    if inputs == None and outputs == None:
        return evaluator(genome)
    elif inputs == None and outputs != None:
        return evaluator(genome,outputs)
    elif inputs != None and outputs == None:
        return evaluator(genome,inputs)
    else:
        return evaluator(genome,inputs,outputs)
        
# Evaluates all genomes in sequential manner (using only 1 process) and
# returns a list of corresponding fitness values.
# evaluator is a callable that is supposed to take Genome as argument and
# return a double
def EvaluateGenomeList_Serial(genome_list, evaluator, inputs=None, outputs=None, display=False):
    fitnesses = []
    count = 0
    curtime = time.time()

    for g in genome_list:
        f = EvaluateGenomeFitness(g,evaluator,inputs,outputs)
        fitnesses.append(f)

        if display:
            print('Individuals: (%s/%s) Fitness: %3.4f' % (count, len(genome_list), f))
        count += 1
        
    elapsed = time.time() - curtime

    if display:
        print('seconds elapsed: %s' % elapsed)

    return fitnesses

# Evaluates all genomes in parallel manner (many processes) and returns a
# list of corresponding fitness values.
# evaluator is a callable that is supposed to take Genome as argument and return a double
def EvaluateGenomeList_Parallel(genome_list, evaluator, inputs=None, outputs=None,cores=-1, display=False):
    curtime = time.time()
    count = 0
    
    if cores == 0:    # run without multiprocessing
        return EvaluateGenomeList_Serial(genome_list,evaluator,inputs,outputs,display)
    elif cores == -1:   # run with the max number of CPUs in the machine
        cores = mp.cpu_count()
    else:              # keep proposed value but ensure it is not above the available CPUs
        cores = min(cores, mp.cpu_count())
    
    if display:
        print('Running on %d cores.' % (cores))
        
    p = mp.Pool(cores)
    threads = []
    for g in genome_list:
        if display:
            print('Starting evaluation for individuals: (%s/%s)' % (count, len(genome_list)))
        args = [g,evaluator,inputs,outputs]
        thread = p.apply_async(EvaluateGenomeFitness,args=args)
        threads.append(thread)
        count += 1
    p.close()                          # do not use p.join as it serializes the processes!
    
    if display:
        print('Waiting for outputs after workers are running.')
        
    try:
        fitnesses = [r.get() for r in threads]
    except KeyError:
        fitnesses = []
    
    if display:
        print('Fitnesses are evaluated.')
    elapsed = time.time() - curtime

    if display:
        print('seconds elapsed: %3.4f' % elapsed)

    return fitnesses



            
            
            
            
            
            
            
            
            
            
            
            
            
