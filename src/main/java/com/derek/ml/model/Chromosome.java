package com.derek.ml.model;


import java.util.List;

public class Chromosome {

    private List<Double> coefficients;
    private double fitnessScore;

    public Chromosome(){}

    public Chromosome(List<Double> coefficients) {
        this.coefficients = coefficients;
    }

    public double getFitnessScore() {
        return fitnessScore;
    }

    public void setFitnessScore(double fitnessScore) {
        this.fitnessScore = fitnessScore;
    }

    public List<Double> getCoefficients() {
        return coefficients;
    }

    public void setCoefficients(List<Double> coefficients) {
        this.coefficients = coefficients;
    }
}
