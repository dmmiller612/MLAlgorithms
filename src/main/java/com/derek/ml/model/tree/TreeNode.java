package com.derek.ml.model.tree;


public class TreeNode {
    private TreeNode left;
    private TreeNode right;
    private Integer attributeIndex;
    private Double splitValue;
    private Double value;

    public TreeNode(){}

    public TreeNode(Double value) {
        this.value = value;
    }

    public TreeNode(TreeNode left, TreeNode right, Integer attributeIndex, Double splitValue) {
        this.left = left;
        this.right = right;
        this.attributeIndex = attributeIndex;
        this.splitValue = splitValue;
    }

    public Double getValue() {
        return value;
    }

    public void setValue(Double value) {
        this.value = value;
    }

    public TreeNode getLeft() {
        return left;
    }

    public void setLeft(TreeNode left) {
        this.left = left;
    }

    public TreeNode getRight() {
        return right;
    }

    public void setRight(TreeNode right) {
        this.right = right;
    }

    public Integer getAttributeIndex() {
        return attributeIndex;
    }

    public void setAttributeIndex(Integer attributeIndex) {
        this.attributeIndex = attributeIndex;
    }

    public Double getSplitValue() {
        return splitValue;
    }

    public void setSplitValue(Double splitValue) {
        this.splitValue = splitValue;
    }
}
