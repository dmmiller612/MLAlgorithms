package com.derek.ml.ro;


import com.derek.ml.model.LabeledPoint;

import java.util.ArrayList;
import java.util.List;

/**
 * TODO
 */
public class SimulatedAnnealing implements RandomizedOptimization {

    double temperature = 1000;
    double coolingRate = .003;
    List<Double> coefficients = null;

    public SimulatedAnnealing(double temperature, double coolingRate){
        this.temperature = temperature;
        this.coolingRate = coolingRate;
    }

    public SimulatedAnnealing(double temperature, double coolingRate, List<Double> coefficients) {
        this.temperature = temperature;
        this.coolingRate = coolingRate;
        this.coefficients = coefficients;
    }

    @Override
    public List<Double> run(List<LabeledPoint> labeledPoints){
        return new ArrayList<>();
    }
}
