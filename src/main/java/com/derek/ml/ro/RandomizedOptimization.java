package com.derek.ml.ro;


import com.derek.ml.model.LabeledPoint;

import java.util.List;

public interface RandomizedOptimization {

    List<Double> run(List<LabeledPoint> labeledPoints);
}
