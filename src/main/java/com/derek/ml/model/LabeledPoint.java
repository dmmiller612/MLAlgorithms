package com.derek.ml.model;


import java.util.List;

public class LabeledPoint {

    public LabeledPoint(){}

    public LabeledPoint(Double outcome, List<Double> predictors) {
        this.outcome = outcome;
        this.predictors = predictors;
    }

    private Double outcome;
    private List<Double> predictors;

    public List<Double> getPredictors() {
        return predictors;
    }

    public void setPredictors(List<Double> predictors) {
        this.predictors = predictors;
    }

    public Double getOutcome() {
        return outcome;
    }

    public void setOutcome(Double outcome) {
        this.outcome = outcome;
    }

}
