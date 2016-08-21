package com.derek.ml.example;


import com.derek.ml.Util;
import com.derek.ml.lib.nb.NaiveBayes;
import com.derek.ml.model.LabeledPoint;

import java.util.List;

public class NaiveBayesExample {

    public static void main(String[] args) {
        List<LabeledPoint> lps = Util.newArrayList(
                new LabeledPoint(1.0, Util.newArrayList(6.0,148.0,72.0,35.,0.,33.6,0.627,50.0)),
                new LabeledPoint(0.0, Util.newArrayList(1.,85.,66.,29.,0.,26.6,0.351,31.)),
                new LabeledPoint(1.0, Util.newArrayList(8.,183.,64.,0.,0.,23.3,0.672,32.)),
                new LabeledPoint(0.0, Util.newArrayList(1.,89.,66.,23.,94.,28.1,0.167,21.)),
                new LabeledPoint(1.0, Util.newArrayList(0.,137.,40.,35.,168.,43.1,2.288,33.)),
                new LabeledPoint(0.0, Util.newArrayList(5.,116.,74.,0.,0.,25.6,0.201,30.)),
                new LabeledPoint(1.0, Util.newArrayList(3.,78.,50.,32.,88.,31.0,0.248,26.)),
                new LabeledPoint(0.0, Util.newArrayList(10.,115.,0.,0.,0.,35.3,0.134,29.)),
                new LabeledPoint(1.0, Util.newArrayList(2.,197.,70.,45.,543.,30.5,0.158,53.)),
                new LabeledPoint(1.0, Util.newArrayList(8.,125.,96.,0.,0.,0.0,0.232,54.)),
                new LabeledPoint(0.0, Util.newArrayList(4.,110.,92.,0.,0.,37.6,0.191,30.)),
                new LabeledPoint(1.0, Util.newArrayList(10.,168.,74.,0.,0.,38.0,0.537,34.))
        );

        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.train(lps);
        for (LabeledPoint lp : lps) {
            System.out.println(naiveBayes.predict(lp.getPredictors()));
        }
    }
}
