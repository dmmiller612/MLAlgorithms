package com.derek.ml.model.ai;


import java.util.List;

public class HiddenLayer {

    private HiddenLayer next = null;
    private List<Node> nodes = null;

    public HiddenLayer(){}

    public HiddenLayer(List<Node> nodes) {
        this.nodes = nodes;
    }

    public boolean isEmpty() {
        return nodes == null || nodes.isEmpty();
    }

    public List<Node> getNodes() {
        return nodes;
    }

    public void setNodes(List<Node> nodes) {
        this.nodes = nodes;
    }

    public HiddenLayer getNext() {
        return next;
    }

    public void setNext(HiddenLayer next) {
        this.next = next;
    }


}
