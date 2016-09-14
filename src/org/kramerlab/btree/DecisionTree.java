package org.kramerlab.btree;

import java.util.HashMap;

import org.kramerlab.shapelets.*;
import org.kramerlab.timeseries.*;

public class DecisionTree {
    public static class Builder {
        // Required params
        private final TimeSeriesDataset trainSet;
        private final int sType;
        
        // Optional params
        private int minLen;
        private int maxLen;
        private int stepSize;
        private int maxTreeDepth;
        private int leafSize;
        
        public Builder(TimeSeriesDataset trainSet, int sType) {
            this.trainSet = trainSet;
            this.minLen = 1;
            this.maxLen = trainSet.get(0).size();
            this.stepSize = 1;
            this.sType = sType;
            this.maxTreeDepth = Integer.MAX_VALUE;
            this.leafSize = 1;
        }
        
        public Builder minLen(int minLen) {
            this.minLen = minLen;
            return this;
        }
        
        public Builder maxLen(int maxLen) {
            this.maxLen = maxLen;
            return this;
        }
        
        public Builder stepSize(int stepSize) {
            this.stepSize = stepSize;
            return this;
        }
        
        public Builder treeDepth(int maxDepth) {
            this.maxTreeDepth = maxDepth;
            return this;
        }
        
        public Builder leafeSize(int leafSize) {
            this.leafSize = leafSize;
            return this;
        }
        
        public DecisionTree build() {
            return new DecisionTree(this);
        }
    }
    
    protected static BaseShapelets shapeletFinder;
    protected static int methodType;
    
    protected int nodeID;
    protected DecisionTree parent;
    protected DecisionTree leftNode;
    protected DecisionTree rightNode;
    protected int leafSize;
    
    protected Shapelet shapelet;
    protected int nodeLabel;
    protected int maxTreeDepth;
    protected int currentTreeDepth;
    
    protected HashMap<Integer, Integer> nodeClassHistogram;
    protected HashMap<Integer, Double> condProb;
    
    protected DecisionTree() {
        this.parent = null;
        this.leftNode = null;
        this.rightNode = null;
        this.shapelet = null;
        this.nodeLabel = Integer.MIN_VALUE;
    }
    
    private DecisionTree(Builder bldr) {
        this();
        methodType = bldr.sType;
        this.nodeID = 1;
        this.currentTreeDepth = 0;
        this.maxTreeDepth = bldr.maxTreeDepth;
        this.leafSize = bldr.leafSize;
        this.condProb = new HashMap<>();
        this.createSubTree(bldr.trainSet, bldr.minLen, bldr.maxLen, bldr.stepSize);
        this.printTree("");
    }
    
    protected DecisionTree(DecisionTree parent, int nodeID, TimeSeriesDataset trainSet, int minLen, int maxLen,
                           int stepSize) {
        this();
        this.nodeID = nodeID;
        this.parent = parent;
        this.currentTreeDepth = parent.currentTreeDepth + 1;
        this.maxTreeDepth = parent.maxTreeDepth;
        this.leafSize = parent.leafSize;
        this.createSubTree(trainSet, minLen, maxLen, stepSize);
    }
    
    protected void createSubTree(TimeSeriesDataset trainSet, int minLen, int maxLen, int stepSize) {
//        System.out.println("Tree Level: " + this.currentTreeDepth + " TrnSet Entropy: " + trainSet.entropy());
        this.nodeClassHistogram = trainSet.getClassHist();
        if (this.currentTreeDepth >= this.maxTreeDepth
                || trainSet.size() <= this.leafSize
                || trainSet.entropy() <= 0.1) {
            this.nodeLabel = trainSet.getClassHist()
                                     .entrySet()
                                     .stream()
                                     .max((x, y) -> x.getValue() > y.getValue() ? 1 : -1)
                                     .get()
                                     .getKey();
            this.updateCondProbability(this.nodeLabel, null);
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
            s += " Classified as: Class " + this.nodeLabel;
            s += " [" + this.nodeClassHistogram + "]";
        }
        System.out.println(s);
        if (this.leftNode != null) {
            this.leftNode.printTree(spaces + "  ");
        }
        if (this.rightNode != null) {
            this.rightNode.printTree(spaces + "  ");
        }
    }
    
    private void updateCondProbability(Integer label, Double prob) {
        if (this.parent == null) {
            this.condProb.put(label,
                              this.condProb.getOrDefault(label, (Double) 0d)
                                     + prob);
        } else {
            int myInstCount = this.nodeClassHistogram.values().stream()
                                                     .mapToInt(e -> e)
                                                     .sum();
            int parentInstCount = this.parent.nodeClassHistogram.values()
                                                                .stream()
                                                                .mapToInt(e -> e)
                                                                .sum();
            double labelProb = (double) myInstCount / parentInstCount;
            
            if (prob != null) {
                labelProb *= prob;
            }
            this.parent.updateCondProbability(label, labelProb);
        }
    }
    
    public HashMap<Integer, Double> getConditionalProbability() {
        return this.condProb;
    }
}
