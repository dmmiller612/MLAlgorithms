package com.derek.ml.example;


import com.derek.ml.Util;
import com.derek.ml.lib.trees.DecisionTree;
import com.derek.ml.lib.trees.DecisionTreeID3;
import com.derek.ml.model.LabeledPoint;

import java.util.List;

public class DecisionTreeExample {

    public static void main(String[] args) {
        List<LabeledPoint> lps = Util.newArrayList(
                new LabeledPoint(1.0, Util.newArrayList(0.0, 0.0, 0.0, 1.0)),
                new LabeledPoint(0.0, Util.newArrayList(0.0, 1.0, 0.0, 1.0)),
                new LabeledPoint(0.0, Util.newArrayList(0.0, 1.0, 1.0, 0.0)),
                new LabeledPoint(1.0, Util.newArrayList(1.0, 1.0, 1.0, 0.0)),
                new LabeledPoint(1.0, Util.newArrayList(1.0, 0.0, 1.0, 0.0)),
                new LabeledPoint(0.0, Util.newArrayList(0.0, 1.0, 1.0, 1.0)),
                new LabeledPoint(1.0, Util.newArrayList(1.0, 1.0, 1.0, 1.0))
        );

        DecisionTree decisionTree = new DecisionTreeID3();
        decisionTree.train(lps);
        for (LabeledPoint lp : lps) {
            System.out.println(decisionTree.predict(lp.getPredictors()));
        }

        List<LabeledPoint> test2 = Util.newArrayList(
                new LabeledPoint(1.0, Util.newArrayList(60.0)),
                new LabeledPoint(1.0, Util.newArrayList(80.0)),
                new LabeledPoint(1.0, Util.newArrayList(59.0)),
                new LabeledPoint(1.0, Util.newArrayList(80.0)),
                new LabeledPoint(1.0, Util.newArrayList(90.0)),
                new LabeledPoint(1.0, Util.newArrayList(66.0)),
                new LabeledPoint(1.0, Util.newArrayList(50.0)),
                new LabeledPoint(1.0, Util.newArrayList(52.0)),
                new LabeledPoint(1.0, Util.newArrayList(73.0)),
                new LabeledPoint(0.0, Util.newArrayList(29.0)),
                new LabeledPoint(0.0, Util.newArrayList(48.0)),
                new LabeledPoint(0.0, Util.newArrayList(33.0)),
                new LabeledPoint(0.0, Util.newArrayList(23.0)),
                new LabeledPoint(0.0, Util.newArrayList(42.0)),
                new LabeledPoint(0.0, Util.newArrayList(10.0)),
                new LabeledPoint(0.0, Util.newArrayList(37.0)),
                new LabeledPoint(0.0, Util.newArrayList(29.0)),
                new LabeledPoint(0.0, Util.newArrayList(40.0))
        );

        DecisionTree decisionTree1 = new DecisionTreeID3();
        decisionTree1.train(test2);
        System.out.println("Break 1");
        for (LabeledPoint lp : test2) {
            System.out.println(decisionTree1.predict(lp.getPredictors()));
        }


        List<LabeledPoint> test3 = Util.newArrayList(
                new LabeledPoint(1.0, Util.newArrayList(0.0, 0.0, 0.0, 1.0, 82.0)),
                new LabeledPoint(0.0, Util.newArrayList(0.0, 1.0, 0.0, 1.0, 35.0)),
                new LabeledPoint(0.0, Util.newArrayList(0.0, 1.0, 1.0, 0.0, 25.0)),
                new LabeledPoint(1.0, Util.newArrayList(1.0, 1.0, 1.0, 0.0, 75.0)),
                new LabeledPoint(1.0, Util.newArrayList(1.0, 0.0, 1.0, 0.0, 66.0)),
                new LabeledPoint(0.0, Util.newArrayList(0.0, 1.0, 1.0, 1.0, 40.0)),
                new LabeledPoint(1.0, Util.newArrayList(1.0, 1.0, 1.0, 1.0, 59.0))
        );

        DecisionTree decisionTree2 = new DecisionTreeID3();
        decisionTree2.train(test3);
        System.out.println("Break 2");
        for (LabeledPoint lp : test3) {
            System.out.println(decisionTree2.predict(lp.getPredictors()));
        }
    }
}
