package com.derek.ml.model;


public class Pair<T, R> {
    public Double one;
    public Double two;

    public T genericOne;
    public R genericTwo;

    public Pair(Double one, Double two){
        this.one = one;
        this.two = two;
    }

    public Pair(T one, R two) {
        this.genericOne = one;
        this.genericTwo = two;
    }

}
