package com.derek.ml.math;


import com.derek.ml.model.LabeledPoint;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class LogFunctions {

    public static double logistic(double value) {
        return 1.0 / (1 + Math.exp(-value));
    }

    public static double logisticPrime(double value) {
        return logistic(value) * (1 - logistic(value));
    }

    public static double logisticLogLikelihood(LabeledPoint labeledPoint, List<Double> beta) {
        if (labeledPoint.getOutcome() == 1.0) {
            return Math.log(logistic(LinearAlgebra.dot(labeledPoint.getPredictors(), beta)));
        } else {
            return Math.log(1 - logistic(LinearAlgebra.dot(labeledPoint.getPredictors(), beta)));
        }
    }

    public static double logisticLogLikelihood(List<LabeledPoint> lps, List<Double> beta) {
        double toReturn = 0;
        for (LabeledPoint lp : lps) {
            toReturn += logisticLogLikelihood(lp, beta);
        }
        return toReturn;
    }

    public static double logisticLogPartial(LabeledPoint lp, List<Double> beta, int x) {
        return (lp.getOutcome() - logistic(LinearAlgebra.dot(lp.getPredictors(), beta))) * lp.getPredictors().get(x);
    }

    public static List<Double> logisticLogGradientX(LabeledPoint lp, List<Double> beta) {
        List<Double> toReturn = new ArrayList<>();
        for (int x = 0; x < beta.size(); x++) {
            toReturn.add(logisticLogPartial(lp, beta, x));
        }
        return toReturn;
    }

    public static List<Double> logisticLogGradient(List<LabeledPoint> lps, List<Double> beta) {
        Optional<List<Double>> opt = lps.stream().map(lp -> logisticLogGradientX(lp, beta)).reduce(LinearAlgebra::vectorAdd);
        return opt.isPresent() ? opt.get() : new ArrayList<>();
    }

}
