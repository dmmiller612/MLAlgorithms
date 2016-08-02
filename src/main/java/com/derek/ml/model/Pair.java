package com.derek.ml.model;


public class Pair<T> {
    public Double one;
    public Double two;

    public T genericOne;
    public T getGenericTwo;

    public Pair(Double one, Double two){
        this.one = one;
        this.two = two;
    }

    public Pair(T one, T two) {
        this.genericOne = one;
        this.getGenericTwo = two;
    }
}
