package org.kramerlab.shapelets;

import java.util.ArrayList;
import java.util.TreeMap;

import org.kramerlab.timeseries.TimeSeries;

public class Shapelet {
    protected TimeSeries shapelet;
    protected double splitDist;
    protected TreeMap<Double, ArrayList<Integer>> dsHist;
    
    public Shapelet() {
        this.shapelet = null;
        this.splitDist = Double.POSITIVE_INFINITY;
        this.dsHist = null;
    }
    
    public Shapelet(TimeSeries shapelet, double splitDist, TreeMap<Double, ArrayList<Integer>> dsHist) {
        this.shapelet = shapelet;
        this.splitDist = splitDist;
        this.dsHist = dsHist;
    }
    
    public TimeSeries getShapelet() {
        return this.shapelet;
    }
    
    public double getSplitDist() {
        return this.splitDist;
    }
    
    public TreeMap<Double, ArrayList<Integer>> getHistMap() {
        return this.dsHist;
    }
}
