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
        TimeSeriesDataset[] splits = new TimeSeriesDataset[2];
        splits[0] = new TimeSeriesDataset();
        splits[1] = new TimeSeriesDataset();
        for (Double d : obj_hist.keySet()) {
            if (d.doubleValue() < split_dist) {
                for (Integer index : obj_hist.get(d)) {
                    splits[0].add(this.trainSet.get(index));
                }
            } else {
                for (Integer index : obj_hist.get(d)) {
                    splits[1].add(this.trainSet.get(index));
                }
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
