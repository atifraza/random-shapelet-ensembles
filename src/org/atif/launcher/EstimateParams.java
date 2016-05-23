package org.atif.launcher;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Properties;

import org.atif.CommonConfig;
import org.kramerlab.shapelets.LegacyShapelets;
import org.kramerlab.shapelets.Shapelet;
import org.kramerlab.timeseries.TimeSeries;
import org.kramerlab.timeseries.TimeSeriesDataset;

public class EstimateParams {
    
    public static void main(String[] args) {
        CommonConfig cc = new CommonConfig(args);
        if (args.length == 0) {
            cc.printHelp(true);
        } else {
            String dsName = cc.getDataSetName();
            String dataPath = cc.getDataPath();
            
            ArrayList<TimeSeries> dataset = cc.loadDataset(Paths.get(dataPath, dsName+"_TRAIN"), " ");
            TimeSeriesDataset trainSet;
            Shapelet currShapelet;
            LegacyShapelets shapeletFinder;
            
            int minLen = 5;
            int maxLen = dataset.get(0).size();
            
            ArrayList<Integer> lens = new ArrayList<Integer>();
            for (int i = 0; i < 100; i++) {
                Collections.shuffle(dataset);
                
                trainSet = new TimeSeriesDataset();
                
                for (int j = 0; j < dataset.size()/4; j++) {  // j < dataset.size()/4
                    trainSet.add(dataset.get(j));
                }
                
                shapeletFinder = new LegacyShapelets(trainSet, minLen, maxLen, 1);
                shapeletFinder.setInversedSearch();
                shapeletFinder.toggleEntropyPruning(false);
                
                currShapelet = shapeletFinder.findShapelet();
                
                added: {
                    for (int j = 0; j < lens.size(); j++) {
                        if (currShapelet.getShapelet().size() < lens.get(j)) {
                            lens.add(j, currShapelet.getShapelet().size());
                            break added;
                        }
                    }
                    lens.add(currShapelet.getShapelet().size());
                }
            }
            
            minLen = lens.get(19);  // 20th entry
            maxLen = lens.get(89);  // 90th entry
            
            Properties props = new Properties();
            props.put("minLen", Integer.toString(minLen));
            props.put("maxLen", Integer.toString(maxLen));
            try (BufferedWriter bw = Files.newBufferedWriter(Paths.get(cc.getParamsPath() + dsName + ".params"))) {
                props.store(bw, "");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
    
}
