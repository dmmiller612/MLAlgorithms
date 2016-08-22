package com.derek.ml.lib.regression;


import com.derek.ml.math.ErrorFunctions;
import com.derek.ml.model.LabeledPoint;
import com.derek.ml.ro.HillClimbing;
import com.derek.ml.ro.RandomizedOptimization;
import com.derek.ml.ro.Target;

import java.util.List;

public class LassoRegression extends Regression {

    protected LassoRegression(List<LabeledPoint> labeledPoints, RandomizedOptimization ro) {
        super(labeledPoints, ro);
    }

    public static LassoRegression lassoRegressionHC(List<LabeledPoint> labeledPoints, int numIterations, double alpha) {
        HillClimbing hillClimbing = new HillClimbing(Target.LASSO, numIterations, alpha);
        return new LassoRegression(addYIntercept(labeledPoints), hillClimbing);
    }

    public Double predict(List<Double> x){
        return ErrorFunctions.predict(x, coefficients);
    }
}
