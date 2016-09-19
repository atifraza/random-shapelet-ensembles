package org.atif.launcher;

import java.util.ArrayList;

import org.atif.CommonConfig;
import org.kramerlab.btree.DecisionTree;
import org.kramerlab.timeseries.TimeSeriesDataset;

public class LaunchRSEnsembleWithBoostingByWeighting {
    
    public static void main(String[] args) {
        CommonConfig cc = new CommonConfig(args, "RS Ensemble - Boosting by Weighting.csv");
        int method = 2;
        TimeSeriesDataset trainSet = new TimeSeriesDataset(cc.getTrainSet()),
                          testSet = new TimeSeriesDataset(cc.getTestSet());
        long start, stop;
        double trainingTime;
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
        trainingTime = (stop - start) / 1e3;
        
        cc.saveResults(dtList, trainSet, testSet, trainingTime);
    }
    
    protected static void normalize(ArrayList<Double> weights) {
        double sum = weights.stream().mapToDouble(val -> val).sum();
        for (int i = 0; i < weights.size(); i++) {
            weights.set(i, weights.get(i)/sum);
        }
    }
}
