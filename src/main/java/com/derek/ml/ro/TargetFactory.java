package com.derek.ml.ro;

import com.derek.ml.math.ErrorFunctions;
import com.derek.ml.math.LogFunctions;
import com.derek.ml.model.LabeledPoint;

import java.util.ArrayList;
import java.util.List;


public class TargetFactory {

    public static double useTarget(Target target, LabeledPoint lp, List<Double> beta, double alpha){
        if (target == Target.SquaredError) {
            return ErrorFunctions.squaredError(lp, beta);
        } else if (target == Target.RIDGE_SQUARE_ERROR) {
            return ErrorFunctions.squaredErrorRidge(lp, beta, alpha);
        } else if (target == Target.NEGATE_LOGISTIC) {
            return ErrorFunctions.negate(LogFunctions.logisticLogLikelihood(lp, beta));
        } else if (target == Target.LASSO) {
             return ErrorFunctions.squaredErrorLasso(lp, beta, alpha);
        }
        return 0;
    }

    public static List<Double> useGradient(Target gradient, LabeledPoint lp, List<Double> beta, double alpha) {
        if (gradient == Target.SquaredError) {
            return ErrorFunctions.squaredErrorGradient(lp.getPredictors(), lp.getOutcome(), beta);
        } else if (gradient == Target.RIDGE_SQUARE_ERROR) {
            return ErrorFunctions.squaredErrorRidgeGradient(lp, beta, alpha);
        } else if (gradient == Target.NEGATE_LOGISTIC) {
            return ErrorFunctions.negateAll(LogFunctions.logisticLogGradientX(lp, beta));
        }
        return new ArrayList<>();
    }
}
