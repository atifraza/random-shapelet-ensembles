package org.kramerlab.shapelets;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Properties;
import java.util.Map.Entry;
import java.util.TreeMap;

import org.kramerlab.timeseries.*;

public class LegacyShapelets extends BaseShapelets {
    
    protected boolean hasMoreCandidates;
    protected boolean entropyPruningEnabled;
    protected boolean decreasingLengthOrder;
    protected boolean normalizationEnabled;
    protected int currInst;
    protected int currPosInInst;
    protected int currLen;
    
    public LegacyShapelets(TimeSeriesDataset trainSet, int minLen, int maxLen, int stepSize) {
        super(trainSet, minLen, maxLen, stepSize);
        Properties props;
        try {
            props = new Properties();
            String propsFileName = System.getProperty("ls-props", "ls.properties");
            props.load(new FileInputStream(propsFileName));
            this.normalizationEnabled = Boolean.parseBoolean(props.getProperty("normalize", "false"));
            this.entropyPruningEnabled = Boolean.parseBoolean(props.getProperty("entropy_pruning", "true"));
            this.decreasingLengthOrder = Boolean.parseBoolean(props.getProperty("decreasing_candidate_length", "true"));
        } catch (Exception e) {
            e.printStackTrace();
        }
        this.hasMoreCandidates = true;
        this.currInst = 0;
        this.currPosInInst = 0;
        this.currLen = this.decreasingLengthOrder ? this.maxLen : this.minLen;
    }
    
    @Override
    public Shapelet findShapelet() {
        TimeSeries bsfShapelet = null;
        TimeSeries currCandidate;
        TimeSeries t;
        double bsfGain = Double.NEGATIVE_INFINITY;
        double bsfSplit = Double.NEGATIVE_INFINITY;
        double currDist;
        TreeMap<Double, ArrayList<Integer>> bsfOrderLine = null;
        TreeMap<Double, ArrayList<Integer>> orderLine = null;
        boolean pruned;
        while ((currCandidate = this.getNextCandidate()) != null) {
            totalCandidates++;
            pruned = false;
            info.gain = Double.NEGATIVE_INFINITY;
            info.splitDist = Double.NEGATIVE_INFINITY;
            // Checking Candidate
            if (orderLine == bsfOrderLine) {
                orderLine = new TreeMap<>();
            } else {
                orderLine.clear();
            }
            for (int ind = 0; ind < this.trainSet.size(); ind++) {
                t = this.trainSet.get(ind);
                currDist = subseqDist(t, currCandidate);
                this.addToMap(orderLine, currDist, ind);
                // Check for early pruning by entropy
                if (this.entropyPruningEnabled && entropyEarlyPruning(orderLine, ind, bsfGain)) {
                    pruned = true;
                    prunedCandidates++;
                    break;
                }
            }
            if (!pruned) {
                this.calcInfoGain_SplitDist(orderLine);
                if (info.gain > bsfGain) {
                    bsfGain = info.gain;
                    bsfSplit = info.splitDist;
                    bsfShapelet = currCandidate;
                    bsfOrderLine = orderLine;
                }
            }
        }
        Shapelet bestFound = new Shapelet(bsfShapelet, bsfSplit, bsfOrderLine);
        return bestFound;
    }
    
    protected TimeSeries getNextCandidate() {
        TimeSeries candidate = null;
        if (this.hasMoreCandidates) {
            TimeSeries currTS = this.trainSet.get(this.currInst);
            candidate = new TimeSeries(currTS, this.currPosInInst, this.currPosInInst + this.currLen);
            this.incrementCandidatePosition();
        }
        return candidate;
    }
    
    protected void incrementCandidatePosition() {
        this.currPosInInst++;
        if (this.currPosInInst + this.currLen > this.trainSet.get(this.currInst).size()) {
            this.currPosInInst = 0;
            this.currInst++;
            if (this.currInst > (this.trainSet.size() - 1)) {
                this.currInst = 0;
                int changeFactor = this.stepSize * (this.decreasingLengthOrder ? -1 : 1);
                this.currLen += changeFactor;
                if ((this.decreasingLengthOrder && (this.currLen < this.minLen))
                    || (!this.decreasingLengthOrder && (this.currLen > this.maxLen))) {
                    this.hasMoreCandidates = false;
                }
            }
        }
    }
    
    protected double subseqDist(TimeSeries t, TimeSeries s) {
        double minDist = Double.POSITIVE_INFINITY;
        boolean stopped;
        double currDist;
        double ti,
               tMu,
               tSigma,
               si,
               sMu = s.mean(0, s.size()),
               sSigma = s.stdv(0, s.size());
        
        for (int tInd = 0; tInd < t.size() - s.size() + 1; tInd++) {
            stopped = false;
            currDist = 0;
            tMu = t.mean(tInd, s.size());
            tSigma = t.stdv(tInd, s.size());
            for (int sInd = 0; sInd < s.size(); sInd++) {
                ti = t.get(tInd + sInd);
                si = s.get(sInd);
                if (this.normalizationEnabled) {
                    ti = (ti - tMu) / tSigma;
                    si = (si - sMu) / sSigma;
                }
                currDist += Math.pow((ti - si), 2);
                if (currDist >= minDist) {
                    stopped = true;
                    break;
                }
            }
            if (!stopped) {
                minDist = currDist;
            }
        }
        return minDist;
    }
    
    protected boolean entropyEarlyPruning(TreeMap<Double, ArrayList<Integer>> orderLine, int ind, double bsfGain) {
        double minEnd = 0;
        double maxEnd = 1 + orderLine.lastKey();
        TreeMap<Double, ArrayList<Integer>> optimisticOrderLine = new TreeMap<>();
        for (Integer cls : this.trainSet.getAllClasses()) {
            createOptimisticOrderLine(optimisticOrderLine, orderLine, ind, cls.intValue(), minEnd, maxEnd);
            this.calcInfoGain_SplitDist(optimisticOrderLine);
            if (info.gain > bsfGain) {
                return false;
            }
            createOptimisticOrderLine(optimisticOrderLine, orderLine, ind, cls.intValue(), maxEnd, minEnd);
            this.calcInfoGain_SplitDist(optimisticOrderLine);
            if (info.gain > bsfGain) {
                return false;
            }
        }
        return true;
    }
    
    private void createOptimisticOrderLine(TreeMap<Double, ArrayList<Integer>> optimisticOrderLine,
                                           TreeMap<Double, ArrayList<Integer>> orderLine, int ind, int cls, double end1,
                                           double end2) {
        optimisticOrderLine.clear();
        for (Entry<Double, ArrayList<Integer>> entry : orderLine.entrySet()) {
            optimisticOrderLine.put(entry.getKey(), new ArrayList<Integer>(entry.getValue()));
        }
        for (int i = ind + 1; i < this.trainSet.size(); i++) {
            if (this.trainSet.get(i).getLabel() == cls) {
                this.addToMap(optimisticOrderLine, end1, i);
            } else {
                this.addToMap(optimisticOrderLine, end2, i);
            }
        }
    }
    
    protected void calcInfoGain_SplitDist(TreeMap<Double, ArrayList<Integer>> orderLine) {
        int size = orderLine.keySet().size();
        Double[] keys = orderLine.keySet().toArray(new Double[size]);
        double meanSplit;
        double currGain;
        double bestGain = Double.NEGATIVE_INFINITY;
        double bestSplit = 0;
        for (int i = 0; i < size - 1; i++) {
            meanSplit = (keys[i] + keys[i + 1]) / 2;
            currGain = this.calcGain(orderLine, meanSplit);
            if (currGain > bestGain) {
                bestGain = currGain;
                bestSplit = meanSplit;
            }
        }
        info.gain = bestGain;
        info.splitDist = bestSplit;
    }
    
    protected double calcGain(TreeMap<Double, ArrayList<Integer>> orderLine, double splitDist) {
        TimeSeriesDataset[] d = splitDataset(orderLine, splitDist);
        return trainSet.entropy() - (d[0].entropy() * d[0].size() + d[1].entropy() * d[1].size()) / trainSet.size();
    }
    
    @Override
    public double getDist(TimeSeries t, TimeSeries s) {
        return subseqDist(t, s);
    }
    
    public void toggleEntropyPruning(boolean newState) {
        this.entropyPruningEnabled = newState;
    }
    
    public void setInversedSearch() {
        this.decreasingLengthOrder = false;
        this.currLen = this.minLen;
    }
    
    public void disableNormalization() {
        this.normalizationEnabled = false;
    }
}
