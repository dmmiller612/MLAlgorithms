package com.derek.ml.lib.regression;


import com.derek.ml.math.ErrorFunctions;
import com.derek.ml.model.LabeledPoint;
import com.derek.ml.ro.RandomizedOptimization;
import com.derek.ml.ro.StochasticGradientDescent;
import com.derek.ml.ro.Target;

import java.util.List;

/**
 * This class is really just a convenience class for ridge multiple regression, which is why the constructor is protected.
 */
public class RidgeRegression extends Regression {

    protected RidgeRegression(List<LabeledPoint> labeledPoints, RandomizedOptimization ro) {
        super(labeledPoints, ro);
    }

    public static RidgeRegression ridgeRegressionSGD(List<LabeledPoint> labeledPoints, int numIterations, double stepSize, double alpha) {
        StochasticGradientDescent stochasticGradientDescent = new StochasticGradientDescent(Target.RIDGE_SQUARE_ERROR, numIterations, stepSize, alpha);
        return new RidgeRegression(addYIntercept(labeledPoints), stochasticGradientDescent);
    }

    public Double predict(List<Double> x){
        return ErrorFunctions.predict(x, coefficients);
    }

}
