package com.derek.ml.lib.nb;


import com.derek.ml.lib.ML;
import com.derek.ml.math.Statistics;
import com.derek.ml.model.LabeledPoint;
import com.derek.ml.model.Pair;

import java.util.ArrayList;
import java.util.List;

public class NaiveBayes implements ML {

    private Pair<List<Pair>, List<Pair>> summaries;

    public void train(List<LabeledPoint> lps) {
        this.summaries = summarize(lps);
    }

    @Override
    public Double predict(List<Double> values) {
        List<Pair> onePair = summaries.genericOne;
        List<Pair> twoPair = summaries.genericTwo;

        double totalProbZero = calculateProb(onePair, values);
        double totalProbOne = calculateProb(twoPair, values);

        if (totalProbZero > totalProbOne) {
            return 0.0;
        } else {
            return 1.0;
        }
    }

    private double calculateProb(List<Pair> values, List<Double> input) {
        double totalProb = 1;
        for (int i = 0; i < values.size(); i++) {
            Double mean = values.get(i).one;
            Double std = values.get(i).two;
            double x = input.get(i);
            totalProb *= Statistics.normalDistribution(x, std, mean);
        }
        return totalProb;
    }

    private Pair<List<LabeledPoint>, List<LabeledPoint>> split(List<LabeledPoint> lps) {
        List<LabeledPoint> zeros = new ArrayList<>();
        List<LabeledPoint> ones = new ArrayList<>();
        for (LabeledPoint lp : lps) {
            if (lp.getOutcome() == 0.0) {
                zeros.add(lp);
            } else {
                ones.add(lp);
            }
        }
        return new Pair<>(zeros, ones);
    }

    private List<Pair> meanStdForAttribute(List<LabeledPoint> lps) {
        List<Pair> toReturn = new ArrayList<>();
        List<List<Double>> items = new ArrayList<>();

        lps.get(0).getPredictors().forEach(ignored -> items.add(new ArrayList<>()));
        for (LabeledPoint lp : lps) {
            for (int i = 0; i < lp.getPredictors().size(); i++) {
                items.get(i).add(lp.getPredictors().get(i));
            }
        }

        for (List<Double> item : items) {
            double mean = Statistics.mean(item);
            double std = Statistics.standardDeviation(item);
            toReturn.add(new Pair(mean, std));
        }
        return toReturn;
    }

    private Pair<List<Pair>, List<Pair>> summarize(List<LabeledPoint> lps) {
        Pair<List<LabeledPoint>, List<LabeledPoint>> values = split(lps);
        List<Pair> zeroUStd = meanStdForAttribute(values.genericOne);
        List<Pair> onesUStd = meanStdForAttribute(values.genericTwo);
        return new Pair<>(zeroUStd, onesUStd);
    }

}
