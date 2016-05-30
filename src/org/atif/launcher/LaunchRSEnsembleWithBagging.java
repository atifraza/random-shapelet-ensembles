package org.atif.launcher;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.atif.CommonConfig;
import org.kramerlab.btree.DecisionTree;
import org.kramerlab.timeseries.TimeSeries;
import org.kramerlab.timeseries.TimeSeriesDataset;

public class LaunchRSEnsembleWithBagging {
    
    public static void main(String[] args) {
        CommonConfig cc = new CommonConfig(args);
        if (args.length == 0) {
            cc.printHelp(true);
        } else {
            String dsName = cc.getDataSetName();
            int method = 2;
            String dataPath = cc.getDataPath();
            String resultsPath = Paths.get(cc.getResultsPath(), "RS Ensemble - Bagging.csv").toString();
            System.out.println("Data set: " + dsName);
            System.out.println("Method: " + method);
            System.out.println();
            ArrayList<TimeSeries> dataset;
            TimeSeriesDataset trainSet, testSet, trainSetBagged;
            int minLen, maxLen, stepSize;
            long start, stop;
            DecisionTree tree;
            int correct = 0;
            int predClass = Integer.MIN_VALUE;
            long totalCandidates = 0, prunedCandidates = 0;
            int ensembleSize = 10;
            Random rng = new Random();
            ArrayList<DecisionTree> dtList = new ArrayList<>();
            Map<Integer, Integer> predClassCount = new HashMap<>();
            int majorityVote;
            IntStream randIntStream;
            List<Integer> randIndices;
            
            dataset = cc.loadDataset(Paths.get(dataPath, dsName + "_TRAIN"), " ");
            trainSet = new TimeSeriesDataset(dataset);
            
            Properties props = cc.constructPropertiesObject(trainSet.get(0).size());
            minLen = Integer.parseInt(props.getProperty("minLen"));
            maxLen = Integer.parseInt(props.getProperty("maxLen"));
            stepSize = Integer.parseInt(props.getProperty("stepSize"));

            start = System.currentTimeMillis();
            for (int i = 0; i<ensembleSize; i++) {
                trainSetBagged = new TimeSeriesDataset();
                randIntStream = rng.ints(dataset.size(), 0, dataset.size());
                randIndices = randIntStream.boxed().collect(Collectors.toList());
                for (Integer ind : randIndices) {
                    trainSetBagged.add(dataset.get(ind));
                }
                tree = new DecisionTree(trainSetBagged, minLen, maxLen, stepSize, method);
                totalCandidates += tree.getTotalCandidates();
                prunedCandidates += tree.getPrunedCandidates();
                dtList.add(tree);
            }
            stop = System.currentTimeMillis();
            
            dataset = cc.loadDataset(Paths.get(dataPath, dsName + "_TEST"), " ");
            testSet = new TimeSeriesDataset(dataset);
            for (int i = 0; i < testSet.size(); i++) {
                predClassCount.clear();
                for (int j = 0; j < ensembleSize; j++) {
                    tree = dtList.get(j);
                    predClass = tree.checkInstance(testSet.get(i));
                    predClassCount.put(predClass, predClassCount.getOrDefault(predClass, 0) + 1);
                }
                majorityVote = predClassCount.entrySet().stream()
                                             .max((e1, e2) -> ((e1.getValue() > e2.getValue()) ? 1 : -1)).get()
                                             .getKey();
                if (majorityVote == testSet.get(i).getLabel()) {
                    correct++;
                }
            }
            
            File resultsFile = new File(resultsPath);
            StringBuilder strBldr = new StringBuilder();
            if (!resultsFile.exists()) {
                strBldr.append("Dataset,TrainSize,TSLen,MinLen,MaxLen,Method,TotalCand,Pruned,Time (sec),Accuracy (%),EnsembleSize\n");
            }
            try (BufferedWriter bw = Files.newBufferedWriter(Paths.get(resultsFile.getAbsolutePath()),
                                                             StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
                strBldr.append(dsName + "," + trainSet.size() + "," + trainSet.get(0).size() + "," + minLen + "," + maxLen
                        + "," + method + "," + totalCandidates + "," + prunedCandidates + ","
                        + (stop - start) / 1e3 + "," + 100.0 * correct / testSet.size() + "," + ensembleSize + "\n");
                bw.write(strBldr.toString());
                System.out.println(strBldr.toString());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
