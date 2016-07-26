package com.derek.ml.ro;


import com.derek.ml.model.LabeledPoint;

import java.util.List;

public interface RandomizedOptimization {

    List<Double> run(Target target, List<LabeledPoint> points, List<Double> theta, int numIterations, double stepSize);
    List<Double> run(Target target, List<LabeledPoint> points, List<Double> theta, int numIterations, double stepSize, double alpha);
}
