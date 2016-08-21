package com.derek.ml.model.ai;


import java.util.List;

public class Node {

    private List<Double> weights;
    private List<Node> connectors;
    private List<Double> inputs;
    private Double bias;

    private Double output;

    public Double getBias() {
        return bias;
    }

    public void setBias(Double bias) {
        this.bias = bias;
    }

    public List<Double> getWeights() {
        return weights;
    }

    public void setWeights(List<Double> weights) {
        this.weights = weights;
    }

    public List<Node> getConnectors() {
        return connectors;
    }

    public void setConnectors(List<Node> connectors) {
        this.connectors = connectors;
    }

    public List<Double> getInputs() {
        return inputs;
    }

    public void setInputs(List<Double> inputs) {
        this.inputs = inputs;
    }

    public Double getOutput() {
        return output;
    }

    public void setOutput(Double output) {
        this.output = output;
    }

}
