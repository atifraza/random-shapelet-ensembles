package org.kramerlab.shapelets;

import java.util.ArrayList;
import java.util.TreeMap;

import org.kramerlab.timeseries.TimeSeries;
import org.kramerlab.timeseries.TimeSeriesDataset;

public abstract class BaseShapelets {
    
    protected static long totalCandidates = 0;
    protected static long prunedCandidates = 0;
    protected TimeSeriesDataset trainSet;
    protected int minLen;
    protected int maxLen;
    protected int stepSize;
    protected Info info;
    
    public BaseShapelets(TimeSeriesDataset trainSet, int minLen, int maxLen, int stepSize) {
        this.trainSet = trainSet;
        this.minLen = minLen;
        this.maxLen = maxLen;
        this.stepSize = stepSize;
        this.info = new Info();
    }
    
    public long getTotalCandidates() {
        return totalCandidates;
    }
    
    public long getPrunedCandidates() {
        return prunedCandidates;
    }
    
    public abstract Shapelet findShapelet();
    
    public abstract double getDist(TimeSeries t, TimeSeries s);
    
    protected void addToMap(TreeMap<Double, ArrayList<Integer>> container, double key, Integer value) {
        ArrayList<Integer> values = container.getOrDefault(key, new ArrayList<Integer>());
        values.add(value);
        container.put(key, values);
    }
    
    public TimeSeriesDataset[] splitDataset(TreeMap<Double, ArrayList<Integer>> obj_hist, double split_dist) {
        final int numSplits = 2;
        final boolean usingWeights = this.trainSet.isUsingWeights();
        
        TimeSeriesDataset[] splits = new TimeSeriesDataset[numSplits];
        double[] sums = new double[numSplits];
        ArrayList<ArrayList<Double>> weights = new ArrayList<>();
        
        for (int i = 0; i < numSplits; i++) {
            splits[i] = new TimeSeriesDataset();
            sums[i] = 0;
            weights.add(new ArrayList<>());
        }
        
        for (Double d : obj_hist.keySet()) {
            if (d.doubleValue() < split_dist) {
                for (Integer index : obj_hist.get(d)) {
                    splits[0].add(this.trainSet.get(index));
                    if (usingWeights) {
                        weights.get(0).add(this.trainSet.getWeight(index));
                        sums[0] += this.trainSet.getWeight(index);
                    }
                }
            } else {
                for (Integer index : obj_hist.get(d)) {
                    splits[1].add(this.trainSet.get(index));
                    if (usingWeights) {
                        weights.get(1).add(this.trainSet.getWeight(index));
                        sums[1] += this.trainSet.getWeight(index);
                    }
                }
            }
        }
        if (usingWeights) {
            for (int i = 0; i < numSplits; i++) {
                for (int j = 0; j < weights.get(i).size(); j++) {
                    weights.get(i).set(j, weights.get(i).get(j) / sums[i]);
                }
                splits[i].setWeights(weights.get(i));
            }
        }
        return splits;
    }
    
    public long getNumOfCandidatesToProcess() {
        long total = 0L;
        long temp;
        for (int cL = this.minLen; cL <= this.maxLen; cL += this.stepSize) {
            temp = 0L;
            for (int cPiI = 0; (cPiI + cL) <= this.trainSet.get(0).size(); cPiI++) {
                temp++;
            }
            total += temp * this.trainSet.size();
        }
        return total;
    }
    
    protected class Info {
        public double gain;
        public double splitDist;
    }
}
