package com.derek.ml.lib.trees;


import com.derek.ml.Util;
import com.derek.ml.model.LabeledPoint;
import com.derek.ml.model.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;


/**
 * Requires binary classification
 */
public class DecisionTreeID3 extends DecisionTree {

    @Override
    protected Pair findBestIndex(List<LabeledPoint> lps){
        double totalEntropy = overallEntropy(lps);
        List<List<Pair>> items = new ArrayList<>();
        lps.get(0).getPredictors().forEach(ignored -> items.add(new ArrayList<>()));
        for (LabeledPoint lp : lps) {
            for (int i = 0; i < lp.getPredictors().size(); i++) {
                items.get(i).add(new Pair(lp.getPredictors().get(i), lp.getOutcome()));
            }
        }

        double bestTotal = Double.NEGATIVE_INFINITY;
        double bestSplit = 1.0;
        int bestIndex = -1;
        for (int i = 0; i < items.size(); i++) {
            Pair entropyPlus = entropyForAttribute(items.get(i));
            double entropy = totalEntropy - entropyPlus.one;
            if (entropy > bestTotal) {
                bestTotal = entropy;
                bestSplit = entropyPlus.two;
                bestIndex = i;
            }
        }
        return new Pair(new Double(bestIndex), bestSplit);
    }

    private double entropy(List<Double> classificationProbabilities) {
        return classificationProbabilities.stream().mapToDouble(prob -> {
            if (prob == 0){
                prob = .001;
            }
            return -prob * Math.log(prob);
        }).sum();
    }

    private double ratio(List<Pair> numbers){
        double total = new Double(numbers.size());
        List<Pair> classOnes = numbers.stream().filter(number -> number.two == 1).collect(Collectors.toList());
        return new Double(classOnes.size()/total);
    }

    private Pair entropyForAttribute(List<Pair> pairs) {
        double splitValue = checkIfContinuous(pairs);
        List<Pair> zeros = pairs.stream().filter(pair -> pair.one < splitValue).collect(Collectors.toList());
        List<Pair> ones = pairs.stream().filter(pair -> pair.one >= splitValue).collect(Collectors.toList());

        double zerosRatio = ratio(zeros);
        double onesRatio = ratio(ones);
        double oneEntropy = entropy(Util.newArrayList(onesRatio));
        double zeroEntropy = entropy(Util.newArrayList(zerosRatio));

        return new Pair(((ones.size() / new Double(pairs.size())) * oneEntropy) - ((zeros.size() / new Double(pairs.size())) * zeroEntropy), splitValue);
    }

    private double overallEntropy(List<LabeledPoint> lps) {
        double total = new Double(lps.size());
        double zero = 0.0;
        double one = 0.0;

        for (LabeledPoint lp : lps){
            if (lp.getOutcome() == 0.0) {
                zero++;
            }
            if (lp.getOutcome() == 1.0) {
                one++;
            }
        }
        return entropy(Util.newArrayList((one/total), (zero/total)));
    }
}
