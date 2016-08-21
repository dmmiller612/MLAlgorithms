package com.derek.ml.ro;


import com.derek.ml.math.ErrorFunctions;
import com.derek.ml.math.Statistics;
import com.derek.ml.model.Chromosome;
import com.derek.ml.model.LabeledPoint;
import com.derek.ml.model.Pair;

import java.util.*;
import java.util.stream.Collectors;


/**
 * Author: Derek Miller
 * Real Coded Genetic Algorithm using SBC
 */
public class RealCodedGeneticAlgorithm implements RandomizedOptimization {

    private double mutation = .001;
    private int poolSize = 200;
    private static final Random random = new Random();
    private double crossoverRate = .7;
    private int numberOfGenerations = 2000;
    private double randomMax = 6.0;
    private double randomMin = -6.0;

    private List<LabeledPoint> lps;
    private static final List<Integer> allowedPoolSizes = Arrays.asList(20, 40, 60, 80, 100, 120, 140, 160, 180, 200);

    public RealCodedGeneticAlgorithm(){}

    public RealCodedGeneticAlgorithm(double mutation, int poolSize, double crossoverRate, int numberOfGenerations) {
        this.mutation = mutation;
        if (!allowedPoolSizes.contains(poolSize)) {
            throw new RuntimeException("Pool Size must be 20, 40, 60, 80, 100, 120, 140, 160, 180, 200");
        }
        this.poolSize = poolSize;
        this.crossoverRate = crossoverRate;
        this.numberOfGenerations = numberOfGenerations;
    }

    public RealCodedGeneticAlgorithm(double mutation, int poolSize, double crossoverRate, int numberOfGenerations, double randomMin, double randomMax) {
        this.mutation = mutation;
        if (!allowedPoolSizes.contains(poolSize)) {
            throw new RuntimeException("Pool Size must be 20, 40, 60, 80, 100, 120, 140, 160, 180, 200");
        }
        this.poolSize = poolSize;
        this.crossoverRate = crossoverRate;
        this.numberOfGenerations = numberOfGenerations;
        this.randomMin = randomMin;
        this.randomMax = randomMax;
    }

    public RealCodedGeneticAlgorithm(int numberOfGenerations) {
        this.numberOfGenerations = numberOfGenerations;
    }

    /**
     * Runs the genetic algorithm
     * @param lps a list of non-null labeled points
     * @return the best found optimal coefficients
     */
    @Override
    public List<Double> run(List<LabeledPoint> lps) {
        this.lps = lps;
        List<Chromosome> startPools = generateInitialPool(lps.get(0).getPredictors().size(), poolSize);
        List<Chromosome> chromosomes = findEndChromosomes(0, startPools);
        Chromosome mostFit = findMostFitChromosome(chromosomes);
        return mostFit.getCoefficients();
    }

    private List<Chromosome> findEndChromosomes(int index, List<Chromosome> chromosomes) {
        if (index < numberOfGenerations) {
            List<Chromosome> parents = findOptimalParents(lps, chromosomes);
            List<Chromosome> children = crossoverAll(parents);
            List<Chromosome> mutatedChildren = mutate(children);

            for (Chromosome chromosome : mutatedChildren) {
                chromosome.setFitnessScore(rSquared(lps, chromosome.getCoefficients()));
            }

            int takers = poolSize / 3;
            int rest = poolSize - takers;

            List<Chromosome> bestChildren = findBestNChildren(mutatedChildren, takers);
            List<Chromosome> newGeneration = generateInitialPool(chromosomes.get(0).getCoefficients().size(), rest);
            newGeneration.addAll(bestChildren);

            return findEndChromosomes(index + 1, newGeneration);
        }
        for (Chromosome chromosome: chromosomes) {
            chromosome.setFitnessScore(rSquared(lps, chromosome.getCoefficients()));
        }
        return chromosomes;
    }

    private List<Chromosome> findBestNChildren(List<Chromosome> chromosomes, int n){
        List<Chromosome> sortedChromosomes = chromosomes.stream().sorted((x, y) -> {
            if (x.getFitnessScore() < y.getFitnessScore()) {
                return 1;
            } else {
                return -1;
            }
        }).collect(Collectors.toList());

        return sortedChromosomes.subList(0, n);
    }

    private Chromosome findMostFitChromosome(List<Chromosome> chromosomes){
        double total = Double.NEGATIVE_INFINITY;
        Chromosome c = null;
        for (Chromosome chromosome : chromosomes) {
            if (chromosome.getFitnessScore() > total) {
                total = chromosome.getFitnessScore();
                c = chromosome;
            }
        }
        return c;
    }

    private List<Chromosome> crossoverAll(List<Chromosome> chromosomes){
        List<Chromosome> returnResult = new ArrayList<>();
        for (int i = chromosomes.size() -1; i > 0; i -= 2) {
            Pair<Chromosome> chromosomePair = crossover(chromosomes.get(i), chromosomes.get(i - 1));
            returnResult.add(chromosomePair.genericOne);
            returnResult.add(chromosomePair.genericTwo);
        }
        return returnResult;
    }

    private Pair<Chromosome> crossover(Chromosome a, Chromosome b){
        List<Double> aCoefficients = a.getCoefficients();
        List<Double> bCoefficients = b.getCoefficients();

        List<Double> xNewCoefficients = new ArrayList<>();
        List<Double> yNewCoefficients = new ArrayList<>();
        for (int i = 0; i < aCoefficients.size(); i++) {
            if (random.nextDouble() < crossoverRate) {
                Pair sbcResults = sbc(aCoefficients.get(i), bCoefficients.get(i));
                xNewCoefficients.add(sbcResults.one);
                yNewCoefficients.add(sbcResults.two);
            } else {
                xNewCoefficients.add(aCoefficients.get(i));
                yNewCoefficients.add(bCoefficients.get(i));
            }
        }
        return new Pair<Chromosome>(new Chromosome(xNewCoefficients), new Chromosome(yNewCoefficients));
    }

    private Pair sbc(double x, double y){
        int n = 2;
        double u = random.nextDouble();
        double b;
        if (u <= .5) {
            b = Math.pow(2.0 * u, 1.0 / (n + 1.0));
        } else {
            b = Math.pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (n + 1.0));
        }
        double xNew = 0.5 * (((1.0 + b) * x) + ((1.0 - b) * y));
        double yNew = 0.5 * (((1.0 - b) * x) + ((1.0 + b) * y));
        return new Pair(xNew, yNew);
    }

    // Returns the selected chromosome based on the weights(probabilities)
    private int rouletteSelect(List<Chromosome> weight) {
        // calculate the total weight
        double weight_sum = 0;
        List<Double> expWeights = new ArrayList<>();
        for(int i=0; i < weight.size(); i++) {
            double exp = Math.exp(weight.get(i).getFitnessScore());
            expWeights.add(exp);
            weight_sum += exp;
        }
        // get a random value
        double value = random.nextDouble() * weight_sum;
        // locate the random value based on the weights
        for(int i=0; i < weight.size(); i++) {
            value -= expWeights.get(i);
            if(value <= 0) return i;
        }
        // only when rounding errors occur
        return weight.size() - 1;
    }


    private List<Chromosome> generateInitialPool(int betaSize, int ps) {
        List<Chromosome> pools = new ArrayList<>();
        for (int i = 0; i < ps; i++) {
            List<Double> tempItem = new ArrayList<>();
            for (int x = 0; x < betaSize; x++) {
               tempItem.add(randomMin + (randomMax - randomMin) * random.nextDouble());
            }
            pools.add(new Chromosome(tempItem));
        }
        return pools;
    }

    private List<Chromosome> findOptimalParents(List<LabeledPoint> lps, List<Chromosome> pools){
        for (Chromosome pool : pools) {
            pool.setFitnessScore(rSquared(lps, pool.getCoefficients()));
        }
        List<Chromosome> rouletteResults = new ArrayList<>();

        for (int x = pools.size()-1; x > 0; x-=2) {
            Chromosome one = pools.get(x);
            Chromosome two = pools.get(x - 1);
            int index = rouletteSelect(Arrays.asList(one, two));
            if (index == 0) {
                rouletteResults.add(one);
            } else {
                rouletteResults.add(two);
            }
        }
        return rouletteResults;
    }

    private List<Chromosome> mutate(List<Chromosome> chromosomes){
        for (Chromosome chromosome : chromosomes) {
            List<Double> coefficients = new ArrayList<>();
            for (int i = 0; i < chromosome.getCoefficients().size(); i++) {
                if (random.nextDouble() <= mutation){
                    coefficients.add(chromosome.getCoefficients().get(i) + (chromosome.getCoefficients().get(i) * .1));
                } else {
                    coefficients.add(chromosome.getCoefficients().get(i));
                }
            }
            chromosome.setCoefficients(coefficients);
        }
        return chromosomes;
    }

    /**
     * Fitness function
     */
    private Double rSquared(List<LabeledPoint> labeledPoints, List<Double> coefficients){
        try {
            double sumOfSquaredErrors = labeledPoints.stream().mapToDouble(item -> ErrorFunctions.squaredError(item, coefficients)).sum();
            return 1.0 - sumOfSquaredErrors / Statistics.tss(labeledPoints.stream().map(item -> item.getOutcome()).collect(Collectors.toList()));
        } catch (Exception e) {
            e.printStackTrace();
        }
        return 0.01;
    }

}