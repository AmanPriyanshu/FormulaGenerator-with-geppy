import numpy as np
import geppy as gep
import inspect
from deap import creator, base, tools

class FormulaGenerator:
	def __init__(self, X, Y, operator_list, gene_length=1, num_genes=1, letter_repr=False):
		self.X = X
		self.Y = Y
		self.operator_list = operator_list
		self.gene_length = gene_length
		self.num_genes = num_genes
		if letter_repr:
			self.pset = gep.PrimitiveSet('Main', input_names=[chr(i+ord('a')) for i in range(self.X.shape[1])])
		else:
			self.pset = gep.PrimitiveSet('Main', input_names=['var_'+str(i) for i in range(self.X.shape[1])])
		for operator in self.operator_list:
			args = inspect.getfullargspec(operator)
			self.pset.add_function(operator, len(args.args))

		creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
		creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMax)

		self.toolbox = gep.Toolbox()
		self.toolbox.register('gene_gen', gep.Gene, pset=self.pset, head_length=self.gene_length)
		self.toolbox.register('individual', creator.Individual, gene_gen=self.toolbox.gene_gen, n_genes=self.num_genes)
		self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
		self.toolbox.register('compile', gep.compile_, pset=self.pset)
		self.toolbox.register('evaluate', self.evaluate)
		self.toolbox.register('select', tools.selRoulette)
		self.toolbox.register('mut_uniform', gep.mutate_uniform, pset=self.pset, ind_pb=2 / (2 * self.gene_length + 1))
		self.toolbox.pbs['mut_uniform'] = 0.1
		self.toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)

		self.stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
		self.stats.register("avg", np.mean)
		self.stats.register("std", np.std)
		self.stats.register("min", np.min)
		self.stats.register("max", np.max)

	def evaluate(self, individual):
		func = self.toolbox.compile(individual)
		n_correct = 0
		for x, y in zip(self.X, self.Y):
			prediction = func(*x)
			if abs(prediction - y)<0.1:
				n_correct += 1
		return n_correct


	def run(self, n_pop=50, n_gen=50):
		pop = self.toolbox.population(n=n_pop)
		hof = tools.HallOfFame(1)
		pop, log = gep.gep_simple(pop, self.toolbox, n_generations=n_gen, n_elites=2, stats=self.stats, hall_of_fame=hof, verbose=True)
		best = hof[0]
		print(best)
		gep.export_expression_tree(best, {i.__name__: i.__name__ for i in self.operator_list}, 'best_tree.png')
	
def generate_multiple_samples(N=1000):
	a = np.random.random(size=N)
	b = np.random.uniform(low=-1, high=1, size=N)
	c = np.random.normal(loc=0.0, scale=2, size=N)
	d = np.random.normal(loc=1.0, scale=2, size=N)
	e = np.random.laplace(loc=3.0, scale=1, size=N)
	f = np.random.laplace(loc=-1.0, scale=3, size=N)
	g = np.random.normal(loc=0.0, scale=6, size=N)
	x = np.stack([a,b,c,d,e,f,g]).T
	y = a+f
	return x, y

def add(a,b):
	return a+b
def sub(a,b):
	return a-b
def mult(a,b):
	return a*b
def div(a,b):
	return a/(b+1e-6)
def neg(a):
	return a

if __name__ == '__main__':
	x, y = generate_multiple_samples()
	operator_list = [add, sub, mult, div]
	fg = FormulaGenerator(x, y, operator_list, letter_repr=True)
	fg.run()

#((c+g)*(d*f))+(f/f)

'''
// to where the error was C:\\Users\\amanp\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\deap\\base.py
	def setValues(self, values):
        if type(values)!=tuple:
            values = (values,)
'''