package com.derek.ml.lib.trees;


import com.derek.ml.lib.ML;
import com.derek.ml.model.LabeledPoint;
import com.derek.ml.model.Pair;
import com.derek.ml.model.tree.TreeNode;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public abstract class DecisionTree implements ML {

    protected TreeNode mainTree;

    protected abstract Pair findBestIndex(List<LabeledPoint> lps);

    /**
     * Scrolls through the tree to find the best answer
     */
    public Double predict(List<Double> lps) {
        List<Double> preds = lps;
        TreeNode tree = mainTree;
        while(tree.getValue() == null) {
            Double value = lps.get(tree.getAttributeIndex());
            preds = removeAttributeFromPredictor(preds, tree.getAttributeIndex());

            if (value  < tree.getSplitValue()) {
                tree = tree.getLeft();
            } else {
                tree = tree.getRight();
            }
        }
        return tree.getValue();
    }

    /**
     * makes the tree then sets the state.
     */
    public void train(List<LabeledPoint> lps) {
        this.mainTree = makeTree(lps);
    }

    /**
     * Makes the tree
     */
    protected TreeNode makeTree(List<LabeledPoint> lps){
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
        if (zero == 0.0) {
            return new TreeNode(1.0);
        } else if (one == 0.0) {
            return new TreeNode(0.0);
        } else if (lps.isEmpty()) {
            return new TreeNode(zero > one ? zero : one);
        }

        Pair p = findBestIndex(lps);
        int bestPredictorIndex = new Double(p.one).intValue();

        //if it is just a stump
        if (lps.get(0).getPredictors().size() < 2){
            return new TreeNode(new TreeNode(0.0), new TreeNode(1.0), bestPredictorIndex, p.two);
        }
        //else handle rest of tree
        Pair<List<LabeledPoint>> pair = partitionWhere(lps, bestPredictorIndex, p.two);
        return new TreeNode(makeTree(pair.genericOne), makeTree(pair.genericTwo), bestPredictorIndex, p.two);
    }

    /**
     * Stateless removal
     */
    protected List<Double> removeAttributeFromPredictor(List<Double> predictors, int attributeIndex) {
        return IntStream.range(0, predictors.size())
                .limit(predictors.size()).filter(i -> i != attributeIndex)
                .mapToObj(i -> predictors.get(i))
                .collect(Collectors.toList());
    }

    protected Pair<List<LabeledPoint>> partitionWhere(List<LabeledPoint> lps, int attributeIndex, double splitValue) {
        List<LabeledPoint> zeros = new ArrayList<>();
        List<LabeledPoint> ones = new ArrayList<>();

        for (LabeledPoint lp : lps) {
            if (lp.getPredictors().get(attributeIndex) < splitValue) {
                zeros.add(new LabeledPoint(lp.getOutcome(), removeAttributeFromPredictor(lp.getPredictors(), attributeIndex)));
            } else {
                ones.add(new LabeledPoint(lp.getOutcome(), removeAttributeFromPredictor(lp.getPredictors(), attributeIndex)));
            }

        }
        return new Pair<>(zeros, ones);
    }

    protected double checkIfContinuous(List<Pair> items){
        boolean isContinuous = false;
        for (Pair item : items) {
            if (item.one != 0.0 && item.one != 1.0) {
                isContinuous = true;
                break;
            }
        }

        if (isContinuous) {
            return findSplit(items);
        } else {
            return 1.0;
        }
    }

    protected double findSplit(List<Pair> xy) {
        List<Pair> sorted = xy.stream().sorted((x, y) -> new Long(Math.round(x.one - y.one)).intValue()).collect(Collectors.toList());
        int index = sorted.size()/2;
        if (index < 3) { //less than three items, just return median
            return sorted.get(index).one;
        }
        return climb(sorted, index);
    }

    private double climb(List<Pair> items, int index){
        int bestIndex = index;
        double bestError = calculate(items, items.get(index).one);
        double leftError = calculate(items, items.get(index -1).one);
        double rightError = calculate(items, items.get(index + 1).one);
        boolean foundBest = bestError < leftError && bestError < rightError;

        while (!foundBest && bestIndex > 0 && bestIndex < items.size() -1) {
            if (rightError < bestError) {
                bestError = rightError;
                bestIndex = bestIndex + 1;
                rightError = calculate(items, bestIndex + 1);
            } else if (leftError < bestError) {
                bestError = leftError;
                bestIndex = bestIndex - 1;
                leftError = calculate(items, bestIndex - 1);
            } else {
                foundBest = true;
            }
        }
        return items.get(bestIndex).one;
    }

    private double calculate(List<Pair> items, double split){
        List<Pair> firstSplit = new ArrayList<>();
        List<Pair> secondSplit = new ArrayList<>();
        for (Pair item : items) {
            if (item.one < split) {
                firstSplit.add(item);
            } else {
                secondSplit.add(item);
            }
        }

        double y1 = firstSplit.stream().mapToDouble(x -> x.two).sum() / firstSplit.size();
        double y2 = secondSplit.stream().mapToDouble(x -> x.two).sum() / secondSplit.size();

        double value1 = firstSplit.isEmpty() ? 0.0 : firstSplit.stream().map(x -> x.two).reduce((x,y) -> Math.pow(y1 - x, 2) + Math.pow(y1 - y, 2)).get();
        double value2 = secondSplit.isEmpty() ? 0.0 : secondSplit.stream().map(x -> x.two).reduce((x,y) -> Math.pow(y2 - x, 2) + Math.pow(y2 - y, 2)).get();
        return value1 + value2;
    }

}
