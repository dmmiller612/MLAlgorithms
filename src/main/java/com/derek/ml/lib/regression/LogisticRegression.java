package com.derek.ml.lib.regression;


import com.derek.ml.lib.ML;
import com.derek.ml.math.LinearAlgebra;
import com.derek.ml.math.LogFunctions;
import com.derek.ml.model.LabeledPoint;
import com.derek.ml.ro.RandomizedOptimization;
import com.derek.ml.ro.StochasticGradientDescent;
import com.derek.ml.ro.Target;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class LogisticRegression implements ML {

    protected List<LabeledPoint> labeledPoints;
    protected RandomizedOptimization ro;
    protected double stepSize;
    protected int iterations;
    protected List<Double> coefficients;

    public LogisticRegression(List<LabeledPoint> labeledPoints, RandomizedOptimization ro, double stepSize, int iterations){
        this.labeledPoints = labeledPoints;
        this.ro = ro;
        this.stepSize = stepSize;
        this.iterations = iterations;

        List<Double> starterCo = labeledPoints.get(0).getPredictors().stream().map(xi -> new Random().nextDouble()).collect(Collectors.toList());
        coefficients = ro.run(Target.NEGATE_LOGISTIC, labeledPoints, starterCo, iterations, stepSize);
    }

    public LogisticRegression(List<LabeledPoint> labeledPoints, double stepSize, int iterations) {
        new LogisticRegression(labeledPoints, new StochasticGradientDescent(), stepSize, iterations);
    }

    public Double predict(List<Double> values) {
        double logPredict = LogFunctions.logistic(LinearAlgebra.dot(coefficients, values));
        return logPredict >= .5 ? 1.0 : 0;
    }

}
