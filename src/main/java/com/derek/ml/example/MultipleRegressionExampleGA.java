package com.derek.ml.example;


import com.derek.ml.lib.regression.MultipleRegression;
import com.derek.ml.lib.regression.PolynomialRegression;
import com.derek.ml.model.LabeledPoint;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MultipleRegressionExampleGA {

    public static void main(String args[]) {
        basicMultipleRegression();
    }

    public static void basicMultipleRegression(){
        System.out.println("BASIC MULTIPLE REGRESSION START");
        List<LabeledPoint> labeledPoints = createLabeledPoints();
        MultipleRegression multipleRegression = MultipleRegression.multipleRegressionGA(labeledPoints);
        System.out.println("r^2 equals: " + multipleRegression.rSquared());
        System.out.println("Prediction equals: " + multipleRegression.predict(Arrays.asList(50.0, 25.0)));
        System.out.println("BASIC MULTIPLE REGRESSION END \n \n");
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
