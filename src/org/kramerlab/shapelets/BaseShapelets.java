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
    
    public BaseShapelets(TimeSeriesDataset trainSet, int minLen, int maxLen,
                         int stepSize) {
        this.trainSet = trainSet;
        this.minLen = minLen;
        this.maxLen = maxLen;
        this.stepSize = stepSize;
    }
    
    public long getTotalCandidates() {
        return totalCandidates;
    }
    
    public long getPrunedCandidates() {
        return prunedCandidates;
    }
    
    public abstract Shapelet findShapelet();
    
    public abstract double getDist(TimeSeries t, TimeSeries s);
    
    protected void addToMap(TreeMap<Double, ArrayList<TimeSeries>> container,
                            double key, TimeSeries value) {
        ArrayList<TimeSeries> values = container.getOrDefault(key,
                                                              new ArrayList<TimeSeries>());
        values.add(value);
        container.put(key, values);
    }
    
    public TimeSeriesDataset[] splitDataset(TreeMap<Double, ArrayList<TimeSeries>> obj_hist,
                                            double split_dist) {
        TimeSeriesDataset[] splits = new TimeSeriesDataset[2];
        splits[0] = new TimeSeriesDataset();
        splits[1] = new TimeSeriesDataset();
        for (Double d : obj_hist.keySet()) {
            if (d.doubleValue() < split_dist) {
                splits[0].add(obj_hist.get(d));
            } else {
                splits[1].add(obj_hist.get(d));
            }
        }
        return splits;
    }
}
