package com.derek.ml.example;


import com.derek.ml.lib.regression.LassoRegression;
import com.derek.ml.lib.regression.MultipleRegression;
import com.derek.ml.model.LabeledPoint;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MultipleRegressionHC {

    public static void main(String args[]) {
        hillClimbingRegression();
        hillClimbingLasso();
    }

    public static void hillClimbingRegression() {
        System.out.println("BASIC Hill Climbing");
        List<LabeledPoint> labeledPoints = createLabeledPoints();
        MultipleRegression multipleRegression = MultipleRegression.multipleRegressionRHC(labeledPoints);
        System.out.println("r^2 equals: " + multipleRegression.rSquared());
        System.out.println("Prediction equals: " + multipleRegression.predict(Arrays.asList(50.0, 25.0)));
        System.out.println("BASIC Random Hill Climbing END \n \n");
    }

    public static void hillClimbingLasso(){
        System.out.println("BASIC Hill Climbing Lasso");
        List<LabeledPoint> labeledPoints = createLabeledPoints();
        LassoRegression multipleRegression = LassoRegression.lassoRegressionHC(labeledPoints, 2000, 2);
        System.out.println("r^2 equals: " + multipleRegression.rSquared());
        System.out.println("Prediction equals: " + multipleRegression.predict(Arrays.asList(50.0, 25.0)));
        System.out.println("BASIC Random Hill Climbing Lasso END \n \n");
    }

    public static List<LabeledPoint> createLabeledPoints(){
        List<LabeledPoint> lps = new ArrayList<>();
        lps.add(new LabeledPoint(30.0, Arrays.asList(10.0, 5.0)));
        lps.add(new LabeledPoint(60.0, Arrays.asList(20.0, 10.0)));
        lps.add(new LabeledPoint(40.0, Arrays.asList(25.0, 15.0)));
        lps.add(new LabeledPoint(90.0, Arrays.asList(30.0, 15.0)));
        lps.add(new LabeledPoint(120.0, Arrays.asList(40.0,20.0)));
        lps.add(new LabeledPoint(20.0, Arrays.asList(5.0, 5.0)));
        return lps;

    }
}
