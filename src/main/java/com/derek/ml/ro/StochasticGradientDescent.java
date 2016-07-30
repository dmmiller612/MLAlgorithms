package com.derek.ml.ro;


import com.derek.ml.math.ErrorFunctions;
import com.derek.ml.math.LinearAlgebra;
import com.derek.ml.math.LogFunctions;
import com.derek.ml.model.LabeledPoint;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class StochasticGradientDescent implements RandomizedOptimization {

    /**
     *
     * @param target this is the target function, such as the squaredError
     * @param points These are the labeled points
     * @param theta These are the initial coefficients or weights that need to be trained
     * @param numIterations number of iterations of no improvement
     * @param stepSize Size of step for gradient
     * @return a list of the best found coefficients
     */
    public List<Double> run(Target target, List<LabeledPoint> points, List<Double> theta, int numIterations, double stepSize, double alpha){
        List<Double> coefficients = theta;
        double step = stepSize <= 0 ? .01 : stepSize;
        List<Double> minTheta = null;
        double minValue = Double.MAX_VALUE;
        double iterationsWithNoImprovement = 0;

        //if there are several iterations with no improvement, we are at least at a local optima
        while (iterationsWithNoImprovement < numIterations) {
            final List<Double> tempCoefficients = coefficients; //here for final usage
            //sum of the points applied to one of the target functions
            double value = points.stream().mapToDouble(lp -> useTarget(target, lp, tempCoefficients, alpha)).sum();
            //if the value is less than min value, we want to go to that point
            if (value < minValue) {
                minTheta = coefficients;
                minValue = value;
                iterationsWithNoImprovement = 0;
                step = stepSize;
            } else {
                //limit stepsize to find areas we may have missed
                iterationsWithNoImprovement += 1;
            }

            //randomize the points
            List<LabeledPoint> randomizedPoints = randomOrder(points);
            for (LabeledPoint lp: randomizedPoints) {
                //get the gradient with the coefficients
                List<Double> grad = useGradient(target, lp, coefficients, alpha);
                //the new coefficients are the coefficients subtracted by the gradient's * step
                coefficients = LinearAlgebra.vectorSubtract(coefficients, LinearAlgebra.scalarMultiply(grad, step));
            }
        }

        return minTheta;
    }

    public List<Double> run(Target target, List<LabeledPoint> points, List<Double> theta, int numIterations, double stepSize) {
        return run(target, points, theta, numIterations, stepSize, 0);
    }

    private List<LabeledPoint> randomOrder(List<LabeledPoint> lps){
        Collections.shuffle(lps);
        return lps;
    }

    private double useTarget(Target target, LabeledPoint lp, List<Double> beta, double alpha){
         if (target == Target.SquaredError) {
            return ErrorFunctions.squaredError(lp, beta);
         } else if (target == Target.RIDGE_SQUARE_ERROR) {
             return ErrorFunctions.squaredErrorRidge(lp, beta, alpha);
         } else if (target == Target.NEGATE_LOGISTIC) {
             return ErrorFunctions.negate(LogFunctions.logisticLogLikelihood(lp, beta));
         }
        return 0;
    }

    private List<Double> useGradient(Target gradient, LabeledPoint lp, List<Double> beta, double alpha) {
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
