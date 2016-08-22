package com.derek.ml.ro;


import com.derek.ml.model.LabeledPoint;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class HillClimbing implements RandomizedOptimization {

    private double acceleration = 1.2;
    private double[] neighbors = new double[]{-acceleration, -1 / acceleration, 0, 1 / acceleration, acceleration};
    private int iterations = 1000;
    private List<Double> stepSizes;
    private List<Double> coefficients;
    private List<LabeledPoint> labeledPoints;
    private double alpha = 2.0;
    private Target target;
    private static final double epsilon = .000000001;

    public HillClimbing(Target target) {
        this.target = target;
    }

    public HillClimbing(Target target, int iterations) {
        this.target = target;
        this.iterations = iterations;
    }

    public HillClimbing(Target target, int iterations, double alpha) {
        this.target = target;
        this.iterations = iterations;
        this.alpha = alpha;
    }

    public List<Double> run(List<LabeledPoint> labeledPoints){
        this.labeledPoints = labeledPoints;
        this.stepSizes = stepSizes == null ? labeledPoints.get(0).getPredictors().stream().map(xi -> 1.0).collect(Collectors.toList()) : stepSizes;
        this.coefficients = coefficients == null ? labeledPoints.get(0).getPredictors().stream().map(xi -> new Random().nextDouble()).collect(Collectors.toList()) : coefficients;
        int iters = 0;
        while (iters < iterations) {
            //initial evaluation
            double initialValue = evaluateResult(coefficients);

            //iterate over each coefficient
            for (int i = 0; i < coefficients.size(); i++) {
                int best = -1;
                double bestScore = Double.POSITIVE_INFINITY;

                for (int j = 0; j < neighbors.length; j++) {
                    coefficients.set(i, coefficients.get(i) + stepSizes.get(i) * neighbors[j]);
                    //evaluate new coefficients with change to xCoefficient
                    double temp = evaluateResult(coefficients);
                    coefficients.set(i, coefficients.get(i) - stepSizes.get(i) * neighbors[j]);
                    //if the temp is the best score, set it
                    if (temp < bestScore) {
                        bestScore = temp;
                        best = j;
                    }
                }

                //this means that no neighbors were better than original, adjust step size
                if (neighbors[best] == 0){
                    stepSizes.set(i, stepSizes.get(i) / acceleration);
                } else {
                    //new better xCoefficient found
                    coefficients.set(i, coefficients.get(i) + stepSizes.get(i) * neighbors[best]);
                    //multiply stepsize by neighbor
                    stepSizes.set(i, stepSizes.get(i) * neighbors[best]);
                }
            }

            //if there hasn't been a change that is greater than a very low number, we converged
            if (Math.abs(evaluateResult(coefficients) - initialValue) < epsilon) {
                return coefficients;
            }
        }
        return this.coefficients;
    }

    private double evaluateResult(List<Double> coeff){
        return labeledPoints.stream().mapToDouble(lp -> TargetFactory.useTarget(target, lp, coeff, 0.0)).sum();
    }
}
