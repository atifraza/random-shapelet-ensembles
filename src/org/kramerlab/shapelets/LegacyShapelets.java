package org.kramerlab.shapelets;

import java.util.ArrayList;
import java.util.Map.Entry;
import java.util.TreeMap;

import org.kramerlab.timeseries.*;

public class LegacyShapelets extends BaseShapelets {
    
    protected boolean hasMoreCandidates;
    protected boolean pruningAllowed = true;
    protected boolean normalOrder = true;
    protected boolean useNormalization = true;
    protected int currInst;
    protected int currPosInInst;
    protected int currLen;
    
    public LegacyShapelets(TimeSeriesDataset trainSet, int minLen, int maxLen,
                           int stepSize) {
        super(trainSet, minLen, maxLen, stepSize);
        this.hasMoreCandidates = true;
        this.currInst = 0;
        this.currPosInInst = 0;
        this.currLen = this.maxLen;
    }
    
    @Override
    public Shapelet findShapelet() {
        TimeSeries bsfShapelet = null;
        TimeSeries currCandidate;
        TimeSeries t;
        double bsfGain = Double.NEGATIVE_INFINITY;
        double bsfSplit = Double.NEGATIVE_INFINITY;
        double currDist;
        double[] currGainAndDist = new double[2];
        TreeMap<Double, ArrayList<TimeSeries>> bsfOrderLine = null;
        TreeMap<Double, ArrayList<TimeSeries>> orderLine = null;
        boolean pruned;
        while ((currCandidate = this.getNextCandidate()) != null) {
            totalCandidates++;
            pruned = false;
            currGainAndDist[0] = Double.NEGATIVE_INFINITY;
            currGainAndDist[1] = Double.NEGATIVE_INFINITY;
            // Checking Candidate
            if (orderLine == bsfOrderLine) {
                orderLine = new TreeMap<Double, ArrayList<TimeSeries>>();
            } else {
                orderLine.clear();
            }
            for (int ind = 0; ind < this.trainSet.size(); ind++) {
                t = this.trainSet.get(ind);
                currDist = subseqDist(t, currCandidate);
                this.addToMap(orderLine, currDist, t);
                // Check for early pruning by entropy
                if (this.pruningAllowed && entropyEarlyPruning(orderLine, ind, bsfGain)) {
                    pruned = true;
                    prunedCandidates++;
                    break;
                }
            }
            if (!pruned) {
                currGainAndDist = calcInfoGain_SplitDist(orderLine);
                if (currGainAndDist[0] > bsfGain) {
                    bsfGain = currGainAndDist[0];
                    bsfSplit = currGainAndDist[1];
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
            candidate = new TimeSeries(currTS, this.currPosInInst,
                                       this.currPosInInst + this.currLen);
            this.currPosInInst++;
            if (this.currPosInInst + this.currLen > currTS.size()) {
                this.currPosInInst = 0;
                this.currInst++;
                if (this.currInst > (this.trainSet.size() - 1)) {
                    this.currInst = 0;
                    int multiplier = this.normalOrder ? 1 : -1;
                    this.currLen -= multiplier * this.stepSize;
                    if ((this.normalOrder && (this.currLen < this.minLen))
                        || (!this.normalOrder && (this.currLen > this.maxLen))) {
                        this.hasMoreCandidates = false;
                    }
                }
            }
        }
        return candidate;
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
                if (this.useNormalization) {
                    ti = ( t.get(tInd + sInd) - tMu ) / tSigma;
                    si = (s.get(sInd)-sMu)/sSigma;
                } else {
                    ti = t.get(tInd + sInd);
                    si = s.get(sInd);
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
    
    protected boolean entropyEarlyPruning(TreeMap<Double, ArrayList<TimeSeries>> orderLine,
                                          int ind, double bsfGain) {
        double minEnd = 0;
        double maxEnd = 1 + orderLine.lastKey();
        TreeMap<Double, ArrayList<TimeSeries>> optimisticOrderLine = new TreeMap<Double, ArrayList<TimeSeries>>();
        for (Integer cls : this.trainSet.getAllClasses()) {
            createOptimisticOrderLine(optimisticOrderLine, orderLine, ind,
                                      cls.intValue(), minEnd, maxEnd);
            if (this.calcInfoGain_SplitDist(optimisticOrderLine)[0] > bsfGain) {
                return false;
            }
            createOptimisticOrderLine(optimisticOrderLine, orderLine, ind,
                                      cls.intValue(), maxEnd, minEnd);
            if (this.calcInfoGain_SplitDist(optimisticOrderLine)[0] > bsfGain) {
                return false;
            }
        }
        return true;
    }
    
    private void createOptimisticOrderLine(TreeMap<Double, ArrayList<TimeSeries>> optimisticOrderLine,
                                           TreeMap<Double, ArrayList<TimeSeries>> orderLine,
                                           int ind, int cls, double end1,
                                           double end2) {
        optimisticOrderLine.clear();
        for (Entry<Double, ArrayList<TimeSeries>> entry : orderLine.entrySet()) {
            optimisticOrderLine.put(entry.getKey(),
                                    new ArrayList<TimeSeries>(entry.getValue()));
        }
        for (int i = ind + 1; i < this.trainSet.size(); i++) {
            if (this.trainSet.get(i).getLabel() == cls) {
                this.addToMap(optimisticOrderLine, end1, this.trainSet.get(i));
            } else {
                this.addToMap(optimisticOrderLine, end2, this.trainSet.get(i));
            }
        }
    }
    
    protected double[] calcInfoGain_SplitDist(TreeMap<Double, ArrayList<TimeSeries>> orderLine) {
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
        return new double[] {bestGain, bestSplit};
    }
    
    protected double calcGain(TreeMap<Double, ArrayList<TimeSeries>> orderLine,
                              double splitDist) {
        TimeSeriesDataset[] d = splitDataset(orderLine, splitDist);
        double gain = this.trainSet.entropy();
        gain -= (d[0].entropy() * d[0].size() + d[1].entropy() * d[1].size())
                / this.trainSet.size();
        return gain;
    }
    
    @Override
    public double getDist(TimeSeries t, TimeSeries s) {
        return subseqDist(t, s);
    }
    
    public void toggleEntropyPruning(boolean newState) {
        this.pruningAllowed = newState;
    }
    
    public void setInversedSearch() {
        this.normalOrder = false;
        this.currLen = this.minLen;
    }
    
    public void disableNormalization() {
        this.useNormalization = false;
    }
}
