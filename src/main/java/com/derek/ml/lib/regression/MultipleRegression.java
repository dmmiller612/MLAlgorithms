package com.derek.ml.lib.regression;


import com.derek.ml.math.ErrorFunctions;
import com.derek.ml.model.LabeledPoint;
import com.derek.ml.ro.*;

import java.util.List;
import java.util.stream.Collectors;

public class MultipleRegression extends Regression {

    /**
     * Regular Multiple Regression
     * @param labeledPoints all of the labeled points
     * @param ro Randomized optimization problem to use
     */
    public MultipleRegression(List<LabeledPoint> labeledPoints, RandomizedOptimization ro){
        super(labeledPoints, ro);
    }

    public static MultipleRegression multipleRegressionSGD(List<LabeledPoint> labeledPoints, int numIterations, double stepSize) {
        List<Double> starterCo = labeledPoints.get(0).getPredictors().stream().map(xi -> .9).collect(Collectors.toList());
        StochasticGradientDescent stochasticGradientDescent = new StochasticGradientDescent(Target.SquaredError, starterCo, numIterations, stepSize);
        return new MultipleRegression(labeledPoints, stochasticGradientDescent);
    }

    public static MultipleRegression multipleRegressionGA(List<LabeledPoint> labeledPoints) {
        RealCodedGeneticAlgorithm realCodedGeneticAlgorithm = new RealCodedGeneticAlgorithm();
        return new MultipleRegression(labeledPoints, realCodedGeneticAlgorithm);
    }

    public static MultipleRegression multipleRegressionRHC(List<LabeledPoint> labeledPoints) {
        HillClimbing randomHillClimbing = new HillClimbing(Target.SquaredError);
        return new MultipleRegression(labeledPoints, randomHillClimbing);
    }

    public Double predict(List<Double> x){
        return ErrorFunctions.predict(x, coefficients);
    }

}
