package com.derek.ml.ai;


import com.derek.ml.math.LinearAlgebra;
import com.derek.ml.math.LogFunctions;
import com.derek.ml.model.ai.HiddenLayer;
import com.derek.ml.model.ai.Network;
import com.derek.ml.model.ai.Node;

import java.util.*;
import java.util.stream.Collectors;


/**
 * TODO THIS IS NOT FINISHED
 */
public class NeuroEvolutionAI {

    private static final Random random = new Random();

    public static void main(String[] args) {
        NeuroEvolutionAI ai = new NeuroEvolutionAI();
        List<Double> a = ai.results(Arrays.asList(4.6, 2.3, 6.3), ai.generateNodes(3, 5, 3));
    }

    public List<Double> results(List<Double> inputs, Network network) {
        LinkedList<HiddenLayer> hiddenLayers = network.getHiddenLayers();
        for (int i = 0; i < hiddenLayers.size(); i++) {
            if (i == 0) {
                List<Node> x = hiddenLayers.get(i).getNodes();
                for (Node n : x) {
                    n.setInputs(inputs);
                    double output = nodeOutput(n);
                    n.setOutput(output);
                }
            } else {
                List<Node> first = hiddenLayers.get(i - 1).getNodes();
                List<Node> second = hiddenLayers.get(i).getNodes();
                for (Node n : second) {
                    n.setInputs(first.stream().map(no -> no.getOutput()).collect(Collectors.toList()));
                    double output = nodeOutput(n);
                    n.setOutput(output);
                }
            }
        }

        return hiddenLayers.getLast().getNodes().stream().map(z -> z.getOutput()).collect(Collectors.toList());
    }

    private double nodeOutput(Node node) {
        return LogFunctions.logistic(LinearAlgebra.dot(node.getInputs(), node.getWeights()));
    }

    public Network generateNodes(int inputLength, int hiddenLayers, int outputs){
        LinkedList<HiddenLayer> nodes = generateHiddenLayers(inputLength, hiddenLayers, outputs);
        return new Network(nodes);
    }

    private List<Double> randomGenerateWeights(double min, double max, int size) {
        List<Double> weights = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            weights.add(randomNumber(min, max));
        }
        return weights;
    }

    private double randomNumber(double min, double max) {
        return min + (max - min) * random.nextDouble();
    }

    private LinkedList<HiddenLayer> generateHiddenLayers(int initialInputLength, int hiddenLayers, int outputs) {
        LinkedList<HiddenLayer> hiddenLayerLink = new LinkedList<>();

        for (int i = 0; i < hiddenLayers; i++) {
            List<Node> nodeHl = new ArrayList<>();
            int length = hiddenLayerLink.isEmpty() ? initialInputLength : random.nextInt(initialInputLength + 1 - 1) + 1;
            int previousNodeLength = hiddenLayerLink.isEmpty() ? initialInputLength : hiddenLayerLink.getLast().getNodes().size();
            for (int o = 0; o < length; o++) {
                Node node = new Node();
                node.setWeights(randomGenerateWeights(-5.0, 5.0, previousNodeLength));
                nodeHl.add(node);
            }
            hiddenLayerLink.add(new HiddenLayer(nodeHl));
        }

        List<Node> h2 = new ArrayList<>();
        for (int x = 0; x < outputs; x++) {
            Node node = new Node();
            node.setWeights(randomGenerateWeights(-5.0, 5.0, hiddenLayerLink.getLast().getNodes().size()));
            h2.add(node);
        }
        hiddenLayerLink.add(new HiddenLayer(h2));

        return hiddenLayerLink;

    }
}
