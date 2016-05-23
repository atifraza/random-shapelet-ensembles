package org.kramerlab.btree;

import org.kramerlab.shapelets.*;
import org.kramerlab.timeseries.*;

public class DecisionTree {
    protected static BaseShapelets shapeletFinder;
    protected static int methodType;
    
    protected int nodeID;
    protected DecisionTree parent;
    protected DecisionTree leftNode;
    protected DecisionTree rightNode;
    
    protected Shapelet shapelet;
    protected int nodeLabel;
    protected int maxDepth;
    protected int currentTreeDepth;
    
    protected DecisionTree() {
        this.parent = null;
        this.leftNode = null;
        this.rightNode = null;
        this.shapelet = null;
        this.nodeLabel = Integer.MIN_VALUE;
    }
    
    public DecisionTree(TimeSeriesDataset trainSet, int minLen, int maxLen, int stepSize, int sType) {
        this(trainSet, minLen, maxLen, stepSize, sType, Integer.MAX_VALUE);
    }
    
    public DecisionTree(TimeSeriesDataset trainSet, int minLen, int maxLen, int stepSize, int sType, int maxDepth) {
        this();
        methodType = sType;
        this.nodeID = 1;
        this.currentTreeDepth = 0;
        this.maxDepth = maxDepth;
        this.createSubTree(trainSet, minLen, maxLen, stepSize);
        this.printTree("");
    }
    
    protected DecisionTree(DecisionTree parent, int nodeID, TimeSeriesDataset trainSet, int minLen, int maxLen,
                           int stepSize) {
        this();
        this.nodeID = nodeID;
        this.parent = parent;
        this.currentTreeDepth = parent.currentTreeDepth + 1;
        this.maxDepth = parent.maxDepth;
        this.createSubTree(trainSet, minLen, maxLen, stepSize);
    }
    
    protected void createSubTree(TimeSeriesDataset trainSet, int minLen, int maxLen, int stepSize) {
//        System.out.println("Tree Level: " + this.currentTreeDepth + " Node ID: " + this.nodeID);
        if (trainSet.entropy() <= 0.1 || this.currentTreeDepth >= this.maxDepth) {
            this.nodeLabel = trainSet.getClassHist()
                                     .entrySet()
                                     .stream()
                                     .max((x, y) -> x.getValue() > y.getValue() ? 1 : -1)
                                     .get()
                                     .getKey();
//            System.out.println("Leaf Node with Class: " + this.nodeLabel);
        } else {
            switch (methodType) {
                case 1:
                    shapeletFinder = new LegacyShapelets(trainSet, minLen, maxLen, stepSize);
                    break;
                case 2:
                    shapeletFinder = new RandomizedLegacyShapelets(trainSet, minLen, maxLen, stepSize);
                    break;
                case 3:
                    shapeletFinder = new LogicalShapelets(trainSet, minLen, maxLen, stepSize);
                    break;
            }
//            long totalCandidates = shapeletFinder.getNumOfCandidatesToProcess();
//            if (shapeletFinder instanceof RandomizedLegacyShapelets) {
//                totalCandidates *= ((RandomizedLegacyShapelets) shapeletFinder).getSamplingPercentage();
//            }
//            System.out.println("Candidates to process at this node: " + totalCandidates);
            this.shapelet = shapeletFinder.findShapelet();
            TimeSeriesDataset[] splitDataset = shapeletFinder.splitDataset(this.shapelet.getHistMap(),
                                                                           this.shapelet.getSplitDist());
            this.leftNode = new DecisionTree(this, 2 * this.nodeID, splitDataset[0], minLen, maxLen, stepSize);
            this.rightNode = new DecisionTree(this, 2 * this.nodeID + 1, splitDataset[1], minLen, maxLen, stepSize);
        }
    }
    
    public int checkInstance(TimeSeries testInst) {
        if (this.nodeLabel != Integer.MIN_VALUE) {
            return this.nodeLabel;
        } else {
            double dist = shapeletFinder.getDist(testInst, this.shapelet.getShapelet());
            if (dist < this.shapelet.getSplitDist()) {
                return this.leftNode.checkInstance(testInst);
            } else {
                return this.rightNode.checkInstance(testInst);
            }
        }
    }
    
    public long getTotalCandidates() {
        return shapeletFinder.getTotalCandidates();
    }
    
    public long getPrunedCandidates() {
        return shapeletFinder.getPrunedCandidates();
    }
    
    protected void printTree(String spaces) {
        String s = spaces + this.nodeID;
        if (this.nodeLabel != Integer.MIN_VALUE) {
            s += " Class Label: " + this.nodeLabel;
        }
        System.out.println(s);
        if (this.leftNode != null) {
            this.leftNode.printTree(spaces + "  ");
        }
        if (this.rightNode != null) {
            this.rightNode.printTree(spaces + "  ");
        }
    }
}
