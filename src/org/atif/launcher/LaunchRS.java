package org.atif.launcher;

import org.atif.CommonConfig;
import org.kramerlab.btree.DecisionTree;
import org.kramerlab.timeseries.TimeSeriesDataset;

public class LaunchRS {
    
    public static void main(String[] args) {
        CommonConfig cc = new CommonConfig(args, "RS.csv");
        int method = 2;
        TimeSeriesDataset trainSet = new TimeSeriesDataset(cc.getTrainSet()),
                          testSet = new TimeSeriesDataset(cc.getTestSet());
        long start, stop;
        double trainingTime, testingTime;
        double trainingAccuracy, testingAccuracy;
        DecisionTree tree;
//        long totalCandidates = 0, prunedCandidates = 0;
        
        start = System.currentTimeMillis();
        tree = new DecisionTree.Builder(trainSet, method)
                               .minLen(cc.getMinLen())
                               .maxLen(cc.getMaxLen())
                               .stepSize(cc.getStepSize())
                               .leafeSize(cc.getLeafSize())
                               .treeDepth(cc.getTreeDepth())
                               .build();
        stop = System.currentTimeMillis();
        trainingAccuracy = getSplitAccuracy(tree, trainSet);
        trainingTime = (stop - start) / 1e3;
        
//        totalCandidates = tree.getTotalCandidates();
//        prunedCandidates = tree.getPrunedCandidates();
        
        start = System.currentTimeMillis();
        testingAccuracy = getSplitAccuracy(tree, testSet);
        stop = System.currentTimeMillis();
        testingTime = (stop - start) / 1e3;
        
        cc.saveResults(trainingTime, testingTime, trainingAccuracy, testingAccuracy);
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
