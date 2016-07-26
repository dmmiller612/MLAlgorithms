package com.derek.ml.lib.regression;


import com.derek.ml.lib.ML;
import com.derek.ml.math.ErrorFunctions;
import com.derek.ml.math.Statistics;
import com.derek.ml.model.LabeledPoint;
import com.derek.ml.ro.RandomizedOptimization;
import com.derek.ml.ro.Target;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class MultipleRegression implements ML {

    protected List<LabeledPoint> labeledPoints;
    protected RandomizedOptimization ro;
    protected double stepSize;
    protected int iterations;
    protected List<Double> coefficients;
    protected double alpha;

    protected MultipleRegression(){}

    /**
     * Regular Multiple Regression
     * @param labeledPoints all of the labeled points
     * @param ro Randomized optimization problem to use
     * @param stepSize stepsize of sgd step
     * @param iterations number of iterations for stochastic Gradient descent
     */
    public MultipleRegression(List<LabeledPoint> labeledPoints, RandomizedOptimization ro, double stepSize, int iterations){
        this.labeledPoints = labeledPoints;
        this.ro = ro;
        this.stepSize = stepSize;
        this.iterations = iterations;

        List<Double> starterCo = labeledPoints.get(0).getPredictors().stream().map(xi -> new Random().nextDouble()).collect(Collectors.toList());
        coefficients = ro.run(Target.SquaredError, labeledPoints, starterCo, iterations, stepSize);
    }

    /**
     * Ridge Regression
     * @param labeledPoints all of the labeled points
     * @param ro Randomized optimization problem to use
     * @param stepSize stepsize of sgd step
     * @param iterations number of iterations for randomized optimization
     * @param alpha the lambda or ridge penalty
     */
    public MultipleRegression(List<LabeledPoint> labeledPoints, RandomizedOptimization ro, double stepSize, int iterations, double alpha) {
        this.labeledPoints = labeledPoints;
        this.ro = ro;
        this.stepSize = stepSize;
        this.iterations = iterations;
        this.alpha = alpha;

        List<Double> starterCo = labeledPoints.get(0).getPredictors().stream().map(xi -> new Random().nextDouble()).collect(Collectors.toList());
        coefficients = ro.run(Target.RIDGE_SQUARE_ERROR, labeledPoints, starterCo, iterations, stepSize, alpha);
    }


    public Double predict(List<Double> x){
        return ErrorFunctions.predict(x, coefficients);
    }

    public Double rSquared(){
        double sumOfSquaredErrors = labeledPoints.stream().mapToDouble(item -> ErrorFunctions.squaredError(item, coefficients)).sum();
        return 1.0 - sumOfSquaredErrors / Statistics.tss(labeledPoints.stream().map(item -> item.getOutcome()).collect(Collectors.toList()));
    }

    public List<LabeledPoint> getLabeledPoints(){
        return labeledPoints;
    }

}
