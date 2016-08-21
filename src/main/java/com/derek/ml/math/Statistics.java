package com.derek.ml.math;


import java.util.List;
import java.util.stream.Collectors;

public class Statistics {

    public static final double pi = 3.141592653589793;

    public static Double mean(List<Double> values) {
        return values.stream().mapToDouble(x -> x).sum() / values.size();
    }

    public static List<Double> deviationMean(List<Double> values) {
        Double m = mean(values);
        return values.stream().map(item -> item - m).collect(Collectors.toList());
    }

    //total sum of squares
    public static Double tss(List<Double> values) {
        return deviationMean(values).stream().mapToDouble(item -> Math.pow(item, 2)).sum();
    }

    public static double variance(List<Double> values) {
        return tss(values) / values.size();
    }

    public static double standardDeviation(List<Double> values) {
        return Math.sqrt(variance(values));
    }

    public static double normalDistribution(double guassValue, double standardDeviation, double mean) {
        return (1 / (Math.sqrt(2 * pi) * standardDeviation)) * Math.exp(-(Math.pow(guassValue - mean, 2) / (2 * Math.pow(standardDeviation, 2))));
    }

}
