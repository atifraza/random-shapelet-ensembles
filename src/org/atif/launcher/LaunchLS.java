package org.atif.launcher;

import org.atif.CommonConfig;
import org.kramerlab.btree.DecisionTree;
import org.kramerlab.timeseries.TimeSeriesDataset;

public class LaunchLS {
    
    public static void main(String[] args) {
        CommonConfig cc = new CommonConfig(args, "LS.csv");
        int method = 1;
        TimeSeriesDataset trainSet = new TimeSeriesDataset(cc.getTrainSet()),
                          testSet = new TimeSeriesDataset(cc.getTestSet());
        long start, stop;
        double trainingTime;
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
        trainingTime = (stop - start) / 1e3;
        
//        totalCandidates = tree.getTotalCandidates();
//        prunedCandidates = tree.getPrunedCandidates();
        
        cc.saveResults(tree, trainSet, testSet, trainingTime);
    }
}
