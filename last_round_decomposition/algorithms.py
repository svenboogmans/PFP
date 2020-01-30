from organism import *
from population import *
import numpy as np
import math
from PIL import Image, ImageOps
import random
import csv

class Algorithm():
    def __init__(self, goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory):
        self.goal = goal
        self.goalpx = np.array(goal)
        self.w = w
        self.h = h
        self.num_poly = num_poly
        self.num_vertex = num_vertex
        self.comparison_method = comparison_method
        self.data = []
        self.savepoints = savepoints
        self.outdirectory = outdirectory

    def save_data(self, row):
        # function to be called every generation to remember data on that generation
        self.data.append(row)

    def write_data(self):
        # writes all collected data to a file
        with open(self.outdirectory + '/data.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            for row in self.data:
                writer.writerow(row)

class Hillclimber(Algorithm):
    def __init__(self, goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory, iterations):
        super().__init__(goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory)
        self.iterations = iterations

        # initializing organism
        self.best = Organism(0,0,None, self.w, self.h, 0)
        self.best.initialize_genome(self.num_poly, num_vertex)
        self.best.genome_to_array()
        self.best.calculate_fitness_mse(self.goalpx)

        # define data header for hillclimber
        self.data.append(["Polygons", "Generation", "MSE"])

    def run(self):
        iterationnr = 0
        #self.best.save_img(self.outdirectory)
        #self.best.save_polygons(self.outdirectory)
        self.save_data([self.num_poly, iterationnr, self.best.fitness])
        for j in range(self.num_poly):
            james = Organism(0, 0, None, self.w, self.h, j)
            james.genome = self.best.deepish_copy_genome()
            james.set_alpha_value()
            james.genome_to_array()
            self.best = deepcopy(james)
            self.best.calculate_fitness_mse(self.goalpx)
            for i in range(0, int((0.75*self.iterations)/self.num_poly)):
                james = Organism(0, 0, None, self.w, self.h, j)
                james.genome = self.best.deepish_copy_genome()
                james.id = iterationnr
                james.generation = iterationnr
                james.random_mutation(1)
                james.genome_to_array()

                if self.comparison_method == "MSE":
                    james.calculate_fitness_mse(self.goalpx)

                if james.fitness <= self.best.fitness:
                    self.best = deepcopy(james)
                    # print(self.best.fitness)
                    # best.save_img()

                iterationnr += 1
                if iterationnr in self.savepoints:
                    #self.best.save_img(self.outdirectory)
                    #self.best.save_polygons(self.outdirectory)
                    self.save_data([self.num_poly, iterationnr, self.best.fitness])

        for i in range(0, int(0.25*self.iterations)):
            james = Organism(0, 0, None, self.w, self.h, 0)
            james.genome = self.best.deepish_copy_genome()
            james.id = i
            james.recomposed_random_mutation(1)
            james.genome_to_array()

            if self.comparison_method == "MSE":
                james.calculate_fitness_mse(self.goalpx)

            if james.fitness <= self.best.fitness:
                self.best = deepcopy(james)
                # print(self.best.fitness)
                # best.save_img()

            if i+1 in self.savepoints:
                #self.best.save_img(self.outdirectory)
                #self.best.save_polygons(self.outdirectory)
                self.save_data([self.num_poly, iterationnr, self.best.fitness])
            iterationnr += 1

        self.best.save_img(self.outdirectory)
        self.best.save_polygons(self.outdirectory)
        self.save_data([self.num_poly, iterationnr, self.best.fitness])


class PPA(Algorithm):
    def __init__(self, goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory, iterations, pop_size, nmax, mmax):
        super().__init__(goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory)
        self.iterations = iterations
        self.pop_size = pop_size
        self.nmax = nmax
        self.mmax = mmax
        self.evaluations = 0
        self.pop = Population(self.pop_size)
        self.best = None
        self.worst = None
        self.full_eva = 0

        # define data header for hillclimber
        self.data.append(["Polygons", "Generation", "Evaluations", "bestMSE", "worstMSE", "medianMSE", "meanMSE"])

        # fill population with random polygon drawings
        for i in range(self.pop_size):
            alex = Organism(0, i, None, self.w, self.h, 0)
            alex.initialize_genome(self.num_poly, self.num_vertex)
            alex.genome_to_array()
            alex.calculate_fitness_mse(self.goalpx)
            self.pop.add_organism(alex)

    def calculate_random_runners(self):
        # if the populations max and min fitness are equal, this function generates random runners and distance for all organisms
        for organism in self.pop.organisms:
            organism.nr = random.randint(1, self.nmax)
            organism.d = random.randint(1, self.mmax)

    def calculate_runners(self):
        # default runner calculation for all organisms in the population
        for organism in self.pop.organisms:
            organism.scale_fitness(self.worst.fitness, self.best.fitness)
            organism.calculate_runners(self.nmax, self.mmax)

    def generation(self, gen, j):
        counter = 0
        for organism in self.pop.organisms[:]:
            for i in range(organism.nr):
                james = Organism(gen, counter, organism.name(), self.w, self.h, j)
                james.genome = organism.deepish_copy_genome()
                james.random_mutation(organism.d)
                james.genome_to_array()
                james.calculate_fitness_mse(self.goalpx)

                self.pop.add_organism(james)
                counter += 1

        self.evaluations += counter
        self.full_eva += counter

    def recomposed_generation(self, gen):
        counter = 0
        for organism in self.pop.organisms[:]:
            for i in range(organism.nr):
                james = Organism(gen, counter, organism.name(), self.w, self.h, 0)
                james.genome = organism.deepish_copy_genome()
                james.recomposed_random_mutation(organism.d)
                james.genome_to_array()
                james.calculate_fitness_mse(self.goalpx)

                self.pop.add_organism(james)
                counter += 1

        self.evaluations += counter
        self.full_eva += counter

    def run(self):
        gen = 1

        for j in range(self.num_poly):
            self.evaluations = 0
            counter = 0
            for organism in self.pop.organisms[:]:
                for i in range(organism.nr):
                    james = Organism(gen, counter, organism.name(), self.w, self.h, j)
                    james.genome = organism.deepish_copy_genome()
                    james.set_alpha_value()
                    james.genome_to_array()
                    james.calculate_fitness_mse(self.goalpx)
                    self.pop.add_organism(james)
                    counter += 1

            while self.evaluations < int((0.75*self.iterations)/self.num_poly):
                self.pop.sort_by_fitness()
                self.best = self.pop.return_best()
                self.worst = self.pop.return_worst()

                if self.best.fitness != self.worst.fitness:
                    self.calculate_runners()
                    self.generation(gen, j)
                else:
                    self.calculate_random_runners()
                    self.generation(gen, j)

                self.pop.eliminate()

                if gen in self.savepoints:
                    #self.best.save_img(self.outdirectory)
                    best, worst, median, mean = self.pop.return_data()
                    self.save_data([self.num_poly, gen, self.full_eva, best.fitness, worst.fitness, median, mean])
                gen += 1

        while self.full_eva < self.iterations:
            self.pop.sort_by_fitness()
            self.best = self.pop.return_best()
            self.worst = self.pop.return_worst()

            if self.best.fitness != self.worst.fitness:
                self.calculate_runners()
                self.recomposed_generation(gen)
            else:
                self.calculate_random_runners()
                self.recomposed_generation(gen)

            self.pop.eliminate()

            if gen in self.savepoints:
                best, worst, median, mean = self.pop.return_data()
                self.save_data([self.num_poly, gen, self.full_eva, best.fitness, worst.fitness, median, mean])
            gen += 1

        self.pop.sort_by_fitness()
        self.best = self.pop.return_best()
        self.best.save_img(self.outdirectory)
        best, worst, median, mean = self.pop.return_data()
        self.save_data([self.num_poly, gen, self.full_eva, best.fitness, worst.fitness, median, mean])
