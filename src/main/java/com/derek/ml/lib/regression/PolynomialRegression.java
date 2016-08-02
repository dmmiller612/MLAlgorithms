package com.derek.ml.lib.regression;


import com.derek.ml.model.LabeledPoint;
import com.derek.ml.ro.RandomizedOptimization;
import com.derek.ml.ro.RealCodedGeneticAlgorithm;
import com.derek.ml.ro.StochasticGradientDescent;
import com.derek.ml.ro.Target;

import java.util.ArrayList;
import java.util.List;

public class PolynomialRegression extends MultipleRegression {

    int degree;

    public PolynomialRegression(List<LabeledPoint> labeledPoints, RandomizedOptimization ro, int degree){
        super(transformToPolynomial(labeledPoints, degree), ro);
        this.degree = degree;
    }

    public static PolynomialRegression polynomialRegressionSGD(List<LabeledPoint> labeledPoints, int numIterations, double stepSize, int degree) {
        StochasticGradientDescent stochasticGradientDescent = new StochasticGradientDescent(Target.SquaredError, numIterations, stepSize);
        return new PolynomialRegression(labeledPoints, stochasticGradientDescent, degree);
    }

    public static PolynomialRegression polynomialRegressionGA(List<LabeledPoint> labeledPoints, int degree) {
        RealCodedGeneticAlgorithm realCodedGeneticAlgorithm = new RealCodedGeneticAlgorithm();
        return new PolynomialRegression(labeledPoints, realCodedGeneticAlgorithm, degree);
    }

    public static List<LabeledPoint> transformToPolynomial(List<LabeledPoint> lps, int degree){
        List<LabeledPoint> toReturn = new ArrayList<>();
        for (LabeledPoint lp : lps){
            List<Double> predictors = lp.getPredictors();
            toReturn.add(new LabeledPoint(lp.getOutcome(), polynomialTransform(predictors, degree)));
        }
        return toReturn;
    }

    public static List<Double> polynomialTransform(List<Double> predictors, int degree){
        List<Double> polynomialPredictors = new ArrayList<>();
        for (double one : predictors) {
            for (int i = 0; i < degree; i++) {
                polynomialPredictors.add(Math.pow(one, i+1));
            }
        }
        return polynomialPredictors;
    }

    public Double predict(List<Double> x){
        return super.predict(polynomialTransform(x, degree));
    }
}
