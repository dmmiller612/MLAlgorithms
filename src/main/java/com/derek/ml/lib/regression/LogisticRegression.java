package com.derek.ml.lib.regression;


import com.derek.ml.lib.ML;
import com.derek.ml.math.LinearAlgebra;
import com.derek.ml.math.LogFunctions;
import com.derek.ml.model.LabeledPoint;
import com.derek.ml.ro.RandomizedOptimization;
import com.derek.ml.ro.StochasticGradientDescent;
import com.derek.ml.ro.Target;

import java.util.List;

public class LogisticRegression extends Regression {

    public LogisticRegression(List<LabeledPoint> labeledPoints, RandomizedOptimization ro){
        this.labeledPoints = labeledPoints;
        this.ro = ro;
        coefficients = ro.run(labeledPoints);
    }

    public static LogisticRegression logisticRegressionSGD(List<LabeledPoint> labeledPoints, int numIterations, double stepSize){
        StochasticGradientDescent gradientDescent = new StochasticGradientDescent(Target.NEGATE_LOGISTIC, numIterations, stepSize);
        return new LogisticRegression(labeledPoints, gradientDescent);
    }

    public Double predict(List<Double> values) {
        double logPredict = LogFunctions.logistic(LinearAlgebra.dot(coefficients, values));
        return logPredict >= .5 ? 1.0 : 0;
    }

}
