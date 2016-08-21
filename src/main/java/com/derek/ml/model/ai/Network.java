package com.derek.ml.model.ai;


import java.util.LinkedList;

public class Network {

    LinkedList<HiddenLayer> hiddenLayers;
    private Double fitnessScore;

    public Network(LinkedList<HiddenLayer> hiddenLayers){
        this.hiddenLayers = hiddenLayers;
    }

    public LinkedList<HiddenLayer> getHiddenLayers() {
        return hiddenLayers;
    }

    public void setHiddenLayers(LinkedList<HiddenLayer> hiddenLayers) {
        this.hiddenLayers = hiddenLayers;
    }

    public Double getFitnessScore() {
        return fitnessScore;
    }

    public void setFitnessScore(Double fitnessScore) {
        this.fitnessScore = fitnessScore;
    }
}
