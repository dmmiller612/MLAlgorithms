package com.derek.ml.ai;


import com.derek.ml.model.LabeledPoint;
import com.derek.ml.model.ai.Network;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * TODO THIS IS NOT FINISHED
 */
public class GeneticAlgorithmNeuro {

    private double mutation = .001;
    private int poolSize = 200;
    private static final Random random = new Random();
    private double crossoverRate = .7;
    private int numberOfGenerations = 1000;
    private double randomMax = 6.0;
    private double randomMin = -6.0;
    private List<LabeledPoint> lps;
    private static final List<Integer> allowedPoolSizes = Arrays.asList(20, 40, 60, 80, 100, 120, 140, 160, 180, 200);
    NeuroEvolutionAI neuroEvolutionAI = new NeuroEvolutionAI();

    public static void main(String args[]) {
        GeneticAlgorithmNeuro n = new GeneticAlgorithmNeuro();
        List<LabeledPoint> labeledPoints = new ArrayList<>();
        for (int i = 0; i < 100; i++) {

        }
    }

    public List<Network> run(List<LabeledPoint> lps) {
        this.lps = lps;
        return generateNetworks();
    }

    private List<Network> generateNetworks() {
        List<Network> parent = generateInitial();

        int i = 0;
        while(i < numberOfGenerations) {
            List<Network> nz = findBestNChildren(parent, 25);
            for (int x = 0; x < poolSize - 25; x++) {
                Network network = neuroEvolutionAI.generateNodes(lps.get(0).getPredictors().size(), random.nextInt(10 + 1 - 1) + 1, 12);
                network.setFitnessScore(fitnessScore(network));
                nz.add(network);
            }
            parent = nz;
            i++;
        }
        return findBestNChildren(parent, 1);
    }

    private List<Network> generateInitial(){
        List<Network> temp = new ArrayList<>();
        for (int x = 0; x < poolSize; x++) {
            Network network = neuroEvolutionAI.generateNodes(lps.get(0).getPredictors().size(), random.nextInt(10 + 1 - 1) + 1, 12);
            network.setFitnessScore(fitnessScore(network));
            temp.add(network);
        }
        return temp;
    }

    private List<Network> findBestNChildren(List<Network> chromosomes, int n){
        List<Network> sortedChromosomes = chromosomes.stream().sorted((x, y) -> {
            if (x.getFitnessScore() < y.getFitnessScore()) {
                return 1;
            } else {
                return -1;
            }
        }).collect(Collectors.toList());

        return sortedChromosomes.subList(0, n);
    }

    private Double fitnessScore(Network network) {
        List<List<Double>> dubs = lps.stream().map(item -> item.getPredictors()).collect(Collectors.toList());
        double reward = 0;
        for (List<Double> dub : dubs) {
            List<Double> result = neuroEvolutionAI.results(dub, network);
            for (int i = 0; i < result.size(); i++) {
                if (result.get(i) > .5) {
                    if (partOfC(i)) {
                        reward += .5;
                    } else {
                        reward -= .5;
                    }
                } else {
                    if (partOfC(i)) {
                        reward -= .09;
                    } else {
                        reward += .09;
                    }

                }
            }
        }
        return reward;
    }

    private boolean partOfC(int i) {
       return (i == 0 || i == 2 || i == 4 || i ==5 || i ==7 || i == 9 || i == 10 || i == 11);
    }

}
