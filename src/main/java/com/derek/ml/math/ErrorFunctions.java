package com.derek.ml.math;

import com.derek.ml.model.LabeledPoint;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;


public class ErrorFunctions {

    public static List<Double> squaredErrorGradient(List<Double> xi, Double yi, List<Double> beta){
        return xi.stream().map(xiJ -> -2 * xiJ * error(yi, xi, beta)).collect(Collectors.toList());
    }

    public static Double squaredError(LabeledPoint lp, List<Double> beta){
        return Math.pow(error(lp.getOutcome(), lp.getPredictors() , beta), 2);
    }

    public static Double error(double yi, List<Double> xi, List<Double> beta){
        return yi - predict(xi, beta);
    }

    public static Double predict(List<Double> xi, List<Double> coefficients){
        return LinearAlgebra.dot(xi, coefficients);
    }

    public static double lassoPenalty(List<Double> coefficients, double alpha) {
        return alpha * coefficients.stream().mapToDouble(item -> Math.abs(item)).sum();
    }

    public static double squaredErrorLasso(LabeledPoint lp, List<Double> beta, double alpha){
        return Math.pow(error(lp.getOutcome(), lp.getPredictors(), beta), 2) + lassoPenalty(beta, alpha);
    }

    /**
     * @param beta are the coefficients
     * @param alpha is the penalty. The lower alpha means the closer to sum of squares. Higher is closer to 0 of coefficients
     * @return penalty
     */
    public static Double ridgePenalty(List<Double> beta, Double alpha) {
        return alpha * LinearAlgebra.dot(beta.subList(1, beta.size()), beta.subList(1, beta.size()));
    }

    public static double squaredErrorRidge(LabeledPoint lp, List<Double> beta, double alpha){
        return Math.pow(error(lp.getOutcome(), lp.getPredictors(), beta), 2) + ridgePenalty(beta, alpha);
    }

    public static List<Double> ridgePenaltyGradient(List<Double> beta, double alpha) {
        List<Double> toReturn = new ArrayList<>();
        toReturn.add(0.0);
        if (beta.size() > 1){
            List<Double> coefficients = beta.subList(1, beta.size());
            for (Double betaI : coefficients) {
                toReturn.add(2 * alpha * betaI);
            }
        }
        return toReturn;
    }

    public static List<Double> squaredErrorRidgeGradient(LabeledPoint lp, List<Double> beta, double alpha){
        return LinearAlgebra.vectorAdd(squaredErrorGradient(lp.getPredictors(), lp.getOutcome(), beta), ridgePenaltyGradient(beta, alpha));
    }

    public static Double negate(Double value){
        return -value;
    }

    public static List<Double> negateAll(List<Double> values){
        return values.stream().map(item -> -item).collect(Collectors.toList());
    }

}
