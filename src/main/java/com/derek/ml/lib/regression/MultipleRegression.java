package com.derek.ml.lib.regression;


import com.derek.ml.lib.ML;
import com.derek.ml.math.ErrorFunctions;
import com.derek.ml.math.Statistics;
import com.derek.ml.model.LabeledPoint;
import com.derek.ml.ro.RandomizedOptimization;
import com.derek.ml.ro.RealCodedGeneticAlgorithm;
import com.derek.ml.ro.StochasticGradientDescent;
import com.derek.ml.ro.Target;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class MultipleRegression implements ML {

    protected List<LabeledPoint> labeledPoints;
    protected RandomizedOptimization ro;
    protected List<Double> coefficients;

    /**
     * Regular Multiple Regression
     * @param labeledPoints all of the labeled points
     * @param ro Randomized optimization problem to use
     */
    public MultipleRegression(List<LabeledPoint> labeledPoints, RandomizedOptimization ro){
        this.labeledPoints = labeledPoints;
        this.ro = ro;
        coefficients = ro.run(labeledPoints);
    }

    public static MultipleRegression multipleRegressionSGD(List<LabeledPoint> labeledPoints, int numIterations, double stepSize) {
        List<Double> starterCo = labeledPoints.get(0).getPredictors().stream().map(xi -> .9).collect(Collectors.toList());
        StochasticGradientDescent stochasticGradientDescent = new StochasticGradientDescent(Target.SquaredError, starterCo, numIterations, stepSize);
        return new MultipleRegression(labeledPoints, stochasticGradientDescent);
    }

    public static MultipleRegression ridgeRegressionSGD(List<LabeledPoint> labeledPoints, int numIterations, double stepSize, double alpha) {
        StochasticGradientDescent stochasticGradientDescent = new StochasticGradientDescent(Target.RIDGE_SQUARE_ERROR, numIterations, stepSize, alpha);
        return new MultipleRegression(addYIntercept(labeledPoints), stochasticGradientDescent);
    }

    public static MultipleRegression multipleRegressionGA(List<LabeledPoint> labeledPoints) {
        RealCodedGeneticAlgorithm realCodedGeneticAlgorithm = new RealCodedGeneticAlgorithm();
        return new MultipleRegression(labeledPoints, realCodedGeneticAlgorithm);
    }

    public Double predict(List<Double> x){
        return ErrorFunctions.predict(x, coefficients);
    }

    public Double rSquared(){
        double sumOfSquaredErrors = labeledPoints.stream().mapToDouble(item -> ErrorFunctions.squaredError(item, coefficients)).sum();
        return 1.0 - sumOfSquaredErrors / Statistics.tss(labeledPoints.stream().map(item -> item.getOutcome()).collect(Collectors.toList()));
    }

    public List<Double> getCoefficients(){
        return coefficients;
    }

    public List<LabeledPoint> getLabeledPoints(){
        return labeledPoints;
    }

    public static List<LabeledPoint> addYIntercept(List<LabeledPoint> lps){
        List<LabeledPoint> toReturn = new ArrayList<>();
        for (LabeledPoint lp : lps) {
            List<Double> predictors = new ArrayList<>();
            predictors.add(1.0);
            predictors.addAll(lp.getPredictors());
            toReturn.add(lp);
        }
        return toReturn;
    }

}
