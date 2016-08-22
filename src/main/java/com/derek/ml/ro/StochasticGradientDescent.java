package com.derek.ml.ro;


import com.derek.ml.math.LinearAlgebra;
import com.derek.ml.model.LabeledPoint;

import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class StochasticGradientDescent implements RandomizedOptimization {

    private Target target;
    private List<Double> theta;
    private int numIterations;
    private double stepSize;
    private double alpha = 0;

    public StochasticGradientDescent(Target target, List<Double> theta, int numIterations, double stepSize, double alpha){
        this.target = target;
        this.theta = theta;
        this.numIterations = numIterations;
        this.stepSize = stepSize;
        this.alpha = alpha;
    }

    public StochasticGradientDescent(Target target, List<Double> theta, int numIterations, double stepSize){
        this.target = target;
        this.theta = theta;
        this.numIterations = numIterations;
        this.stepSize = stepSize;
    }

    public StochasticGradientDescent(Target target, int numIterations, double stepSize, double alpha){
        this.target = target;
        this.numIterations = numIterations;
        this.stepSize = stepSize;
        this.alpha = alpha;
    }

    public StochasticGradientDescent(Target target, int numIterations, double stepSize){
        this.target = target;
        this.numIterations = numIterations;
        this.stepSize = stepSize;
    }

    public StochasticGradientDescent(List<LabeledPoint> lps) {
        this.target = Target.SquaredError;
        this.numIterations = 500;
        this.stepSize = .001;
        this.alpha = 0;
    }

    /**
     *
     * @return a list of the best found coefficients
     */
    @Override
    public List<Double> run(List<LabeledPoint> labeledPoints){
        List<Double> coefficients = theta == null ? labeledPoints.get(0).getPredictors().stream().map(xi -> new Random().nextDouble()).collect(Collectors.toList()) : theta;
        double step = stepSize <= 0 ? .01 : stepSize;
        List<Double> minTheta = null;
        double minValue = Double.MAX_VALUE;
        double iterationsWithNoImprovement = 0;

        //if there are several iterations with no improvement, we are at least at a local optima
        while (iterationsWithNoImprovement < numIterations) {
            final List<Double> tempCoefficients = coefficients; //here for final usage
            //sum of the points applied to one of the target functions
            double value = labeledPoints.stream().mapToDouble(lp -> TargetFactory.useTarget(target, lp, tempCoefficients, alpha)).sum();
            //if the value is less than min value, we want to go to that point
            if (value < minValue) {
                minTheta = coefficients;
                minValue = value;
                iterationsWithNoImprovement = 0;
                step = stepSize;
            } else {
                //limit stepsize to find areas we may have missed
                iterationsWithNoImprovement += 1;
            }

            //randomize the points
            List<LabeledPoint> randomizedPoints = randomOrder(labeledPoints);
            for (LabeledPoint lp: randomizedPoints) {
                //get the gradient with the coefficients
                List<Double> grad = TargetFactory.useGradient(target, lp, coefficients, alpha);
                //the new coefficients are the coefficients subtracted by the gradient's * step
                coefficients = LinearAlgebra.vectorSubtract(coefficients, LinearAlgebra.scalarMultiply(grad, step));
            }
        }

        return minTheta;
    }

    private List<LabeledPoint> randomOrder(List<LabeledPoint> lps){
        Collections.shuffle(lps);
        return lps;
    }


}
