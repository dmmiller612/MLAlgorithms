package com.derek.ml.math;


import java.util.List;
import java.util.stream.Collectors;

public class Statistics {

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

}
