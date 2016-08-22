package com.derek.ml.lib.regression;


import com.derek.ml.lib.ML;
import com.derek.ml.math.ErrorFunctions;
import com.derek.ml.math.Statistics;
import com.derek.ml.model.LabeledPoint;
import com.derek.ml.ro.RandomizedOptimization;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public abstract class Regression implements ML {

    protected List<LabeledPoint> labeledPoints;
    protected RandomizedOptimization ro;
    protected List<Double> coefficients;

    protected Regression(List<LabeledPoint> labeledPoints, RandomizedOptimization ro){
        this.labeledPoints = labeledPoints;
        this.ro = ro;
        coefficients = ro.run(labeledPoints);
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
