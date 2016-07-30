package com.derek.ml.math;

import com.derek.ml.model.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class LinearAlgebra {

    public static double dot(List<Double> one, List<Double> two) {
        if (one.size() != two.size() ) {
            throw new RuntimeException("both lists must be the same size for dot product");
        }
        int size = one.size();
        double totalSum = 0;
        for (int i = 0; i < size; i++) {
            totalSum += (one.get(i) * two.get(i));
        }
        return totalSum;
    }

    public static List<Pair> zip(List<Double> x, List<Double> y){
        int size = x.size();
        List<Pair> pairs = new ArrayList<>();
        for (int i = 0; i < size; i++){
            pairs.add(new Pair(x.get(i), y.get(i)));
        }
        return pairs;
    }

    public static List<Double> vectorSubtract(List<Double> x, List<Double> y) {
        return LinearAlgebra.zip(x, y).stream().map(item -> item.one - item.two).collect(Collectors.toList());
    }

    public static List<Double> scalarMultiply(List<Double> items, Double scalar){
        return items.stream().map(item -> item * scalar).collect(Collectors.toList());
    }

    public static List<Double> vectorAdd(List<Double> one, List<Double> two) {
        return zip(one, two).stream().map(item -> item.one + item.two).collect(Collectors.toList());
    }
}
