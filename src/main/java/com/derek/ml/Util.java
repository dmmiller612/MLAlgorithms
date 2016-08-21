package com.derek.ml;

import java.util.ArrayList;
import java.util.List;


public class Util {

    public static <T> List<T> newArrayList(T ... args) {
        List<T> toReturn = new ArrayList<>();
        if (args != null) {
            for (T arg : args) {
                toReturn.add(arg);
            }
        }
        return toReturn;
    }
}
