package org.atif.launcher;

import java.util.ArrayList;
import java.util.HashMap;

import org.atif.CommonConfig;
import org.kramerlab.btree.DecisionTree;
import org.kramerlab.timeseries.TimeSeriesDataset;

public class LaunchRSEnsembleWithBoosting2 {
    
    public static void main(String[] args) {
        CommonConfig cc = new CommonConfig(args);
        int method = 2;
        TimeSeriesDataset trainSet = new TimeSeriesDataset(cc.getTrainSet()),
                          testSet = new TimeSeriesDataset(cc.getTestSet());
        long start, stop;
        double trainingTime, testingTime;
        double trainingAccuracy, testingAccuracy;
        DecisionTree tree;
//        long totalCandidates = 0, prunedCandidates = 0;
        ArrayList<DecisionTree> dtList = new ArrayList<>();
        ArrayList<Integer> incorrectlyClassified = new ArrayList<>();
        ArrayList<Double> weights = new ArrayList<>();
        int predClass = Integer.MIN_VALUE;
        double error;
        double temp = 1.0 / trainSet.size();
        for (int i = 0; i < trainSet.size(); i++) {
            weights.add(temp);
        }
        
        start = System.currentTimeMillis();
        for (int i = 0; i < cc.getEnsembleSize(); i++) {
            trainSet.setWeights(weights);
            tree = new DecisionTree.Builder(trainSet, method)
                                   .minLen(cc.getMinLen())
                                   .maxLen(cc.getMaxLen())
                                   .stepSize(cc.getStepSize())
                                   .leafeSize(cc.getLeafSize())
                                   .treeDepth(cc.getTreeDepth())
                                   .build();
            error = 0.0;
            for (int j = 0; j < trainSet.size(); j++) {
                predClass = tree.checkInstance(trainSet.get(j));
                if (predClass != trainSet.get(j).getLabel()) {
                    error += weights.get(j);
                    incorrectlyClassified.add(j);
                }
            }
            // totalCandidates += tree.getTotalCandidates();
            // prunedCandidates += tree.getPrunedCandidates();
            dtList.add(i, tree);
//            if (error < 1e-3) {
//                break;
//            }
            temp = (2 * error);
            for (Integer ind : incorrectlyClassified) {
                weights.set(ind, weights.get(ind) / temp);
            }
            incorrectlyClassified.clear();
            normalize(weights);
        }
        stop = System.currentTimeMillis();
        trainingAccuracy = getSplitAccuracy(dtList, trainSet);
        trainingTime = (stop - start) / 1e3;
        
        start = System.currentTimeMillis();
        testingAccuracy = getSplitAccuracy(dtList, testSet);
        stop = System.currentTimeMillis();
        testingTime = (stop - start) / 1e3;
        
        cc.saveResults("RS Ensemble - Boosting2.csv", trainingTime, testingTime, trainingAccuracy, testingAccuracy,
                       dtList.size());
    }
    
    public static double getSplitAccuracy(ArrayList<DecisionTree> dtList, TimeSeriesDataset split) {
        int predClass, correct = 0, majorityVote;
        HashMap<Integer, Integer> predClassCount = new HashMap<>();
        for (int ind = 0; ind < split.size(); ind++) {
            predClassCount.clear();
            for (int j = 0; j < dtList.size(); j++) {
                predClass = dtList.get(j).checkInstance(split.get(ind));
                predClassCount.put(predClass, predClassCount.getOrDefault(predClass, 0) + 1);
            }
            majorityVote = predClassCount.entrySet().stream()
                                         .max((e1, e2) -> ((e1.getValue() > e2.getValue()) ? 1 : -1)).get().getKey();
            if (majorityVote == split.get(ind).getLabel()) {
                correct++;
            }
        }
        return 100.0 * correct / split.size();
    }
    
    protected static void normalize(ArrayList<Double> weights) {
        double sum = weights.stream().mapToDouble(val -> val).sum();
        for (int i = 0; i < weights.size(); i++) {
            weights.set(i, weights.get(i)/sum);
        }
    }
}
