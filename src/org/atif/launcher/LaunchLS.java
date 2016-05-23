package org.atif.launcher;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Properties;

import org.atif.CommonConfig;
import org.kramerlab.btree.DecisionTree;
import org.kramerlab.timeseries.TimeSeries;
import org.kramerlab.timeseries.TimeSeriesDataset;

public class LaunchLS {
    
    public static void main(String[] args) {
        CommonConfig cc = new CommonConfig(args);
        if (args.length == 0) {
            cc.printHelp(true);
        } else {
            String dsName = cc.getDataSetName();
            int method = 1;
            String dataPath = cc.getDataPath();
            String resultsPath = cc.getResultsPath() + "LS.csv";
            System.out.println("Data set: " + dsName);
            System.out.println("Method: " + method);
            System.out.println();
            ArrayList<TimeSeries> dataset;
            TimeSeriesDataset trainSet, testSet;
            int minLen, maxLen, stepSize;
            long start, stop;
            DecisionTree tree;
            int correct = 0;
            int predClass = Integer.MIN_VALUE;
            long totalCandidates = 0, prunedCandidates = 0;
            Properties props = new Properties();
            
            try {
                File propsFile = new File(cc.getParamsPath() + "default.params");
                props.load(new FileInputStream(propsFile));
                propsFile = new File(cc.getParamsPath() + dsName + ".params");
                if (propsFile.exists()) {
                    props.load(new FileInputStream(propsFile));
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            
            dataset = cc.loadDataset(Paths.get(dataPath, dsName + "_TRAIN"), " ");
            trainSet = new TimeSeriesDataset(dataset);
            
            if (props.containsKey("minLen")) {
                minLen = Integer.parseInt(props.getProperty("minLen"));
            } else {
                minLen = (int) Math.ceil(trainSet.get(0).size()/4.0);
            }
            if (props.containsKey("maxLen")) {
                maxLen = Integer.parseInt(props.getProperty("maxLen", "20"));
            } else {
                maxLen = (int) Math.floor(trainSet.get(0).size()*2/3.0);
            }
            stepSize = Integer.parseInt(props.getProperty("stepSize", "1"));
            
            start = System.currentTimeMillis();
            tree = new DecisionTree(trainSet, minLen, maxLen, stepSize, method);
            stop = System.currentTimeMillis();
            totalCandidates = tree.getTotalCandidates();
            prunedCandidates = tree.getPrunedCandidates();
            dataset = cc.loadDataset(Paths.get(dataPath, dsName + "_TEST"), " ");
            testSet = new TimeSeriesDataset(dataset);
            for (int ind = 0; ind < testSet.size(); ind++) {
                predClass = tree.checkInstance(testSet.get(ind));
                if (predClass == testSet.get(ind).getLabel()) {
                    correct++;
                }
            }
            
            File resultsFile = new File(resultsPath);
            String temp = "", temp2;
            if (!resultsFile.exists()) {
                temp = "Dataset,TrainSize,TSLen,MinLen,MaxLen,Method,TotalCand,Pruned,Time (sec),Accuracy (%)\n";
            }
            try (BufferedWriter bw = Files.newBufferedWriter(Paths.get(resultsFile.getAbsolutePath()),
                                                             StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
                temp2 = dsName + "," + trainSet.size() + "," + trainSet.get(0).size() + "," + minLen + "," + maxLen
                        + "," + method + "," + totalCandidates + "," + prunedCandidates + ","
                        + (stop - start) / 1e3 + "," + 100.0 * correct / testSet.size() + "\n";
                bw.write(temp+temp2);
                System.out.println("\n\n" + temp2);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
