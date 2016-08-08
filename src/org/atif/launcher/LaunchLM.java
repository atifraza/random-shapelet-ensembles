package org.atif.launcher;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import org.atif.CommonConfig;
import org.kramerlab.btree.DecisionTree;
import org.kramerlab.timeseries.TimeSeries;
import org.kramerlab.timeseries.TimeSeriesDataset;

public class LaunchLM {
    
    public static void main(String[] args) {
        CommonConfig cc = new CommonConfig(args, "LM.csv");
        int method = 1;
        TimeSeriesDataset originalTrainSet = new TimeSeriesDataset(cc.getTrainSet()),
                originalTestSet = new TimeSeriesDataset(cc.getTestSet()),
                ovaTrainSet, ovaTestSet;
        
        long start, stop;
        HashMap<Integer, DecisionTree> tree = new HashMap<>();
        
        System.out.println("Class distributions:");
        System.out.println("Training: "+ originalTrainSet.getClassHist() + " Testing: " + originalTestSet.getClassHist());
        System.out.println();
        
        TimeSeries t, newSeries;
        start = System.currentTimeMillis();
        for (Integer cls : originalTrainSet.getAllClasses()) {
//            System.out.println("Training model for class " + cls + " against rest.");
            ovaTrainSet = new TimeSeriesDataset();
            for (int i = 0; i < originalTrainSet.size(); i++) {
                t = originalTrainSet.get(i);
                newSeries = new TimeSeries(new ArrayList<Double>(Arrays.asList(t.getTS())), t.getLabel() == cls ? cls : Integer.MAX_VALUE);
                ovaTrainSet.add(newSeries);
            }
            
            ovaTestSet = new TimeSeriesDataset();
            for (int i = 0; i < originalTestSet.size(); i++) {
                t = originalTestSet.get(i);
                if (t.getLabel() == cls) {
                    newSeries = new TimeSeries(new ArrayList<Double>(Arrays.asList(t.getTS())), t.getLabel());
                    ovaTestSet.add(newSeries);
                }
            }
            
//            System.out.println();
//            System.out.println("OVA Class distributions:");
//            System.out.println("Training: "+ ovaTrainSet.getClassHist() + " Testing: " + ovaTestSet.getClassHist());
//            System.out.println();
            tree.put(cls, new DecisionTree.Builder(ovaTrainSet, method)
                                          .minLen(cc.getMinLen())
                                          .maxLen(cc.getMaxLen())
                                          .stepSize(cc.getStepSize())
                                          .leafeSize(cc.getLeafSize())
                                          .treeDepth(1)
                                          .build());
//            System.out.println("OVA Test set classification accuracy: " + getSplitAccuracy(tree.get(cls), ovaTestSet) + "\n\n");
        }
        stop = System.currentTimeMillis();
        System.out.println("Training time: " + (stop - start)/1000.0);
        int predClass = Integer.MIN_VALUE, correct = 0, temp;
        for (int i = 0; i < originalTestSet.size(); i++) {
            t = originalTestSet.get(i);
            for (Integer cls : originalTestSet.getAllClasses()) {
                temp = tree.get(cls).checkInstance(t);
                if (temp == cls) {
                    predClass = temp;
                }
            }
            if (predClass == t.getLabel()) {
                correct++;
            }
        }
        System.out.println("Overall accuracy: " + 100.0*correct/originalTestSet.size());
    }
    
    public static double getSplitAccuracy(DecisionTree tree, TimeSeriesDataset split) {
        int predClass, correct = 0;
        for (int ind = 0; ind < split.size(); ind++) {
            predClass = tree.checkInstance(split.get(ind));
            if (predClass == split.get(ind).getLabel()) {
                correct++;
            }
        }
        return 100.0 * correct / split.size();
    }
}
