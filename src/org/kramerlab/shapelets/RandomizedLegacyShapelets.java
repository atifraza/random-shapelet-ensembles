/**
 * 
 */
package org.kramerlab.shapelets;

import java.io.FileInputStream;
import java.util.Properties;
import java.util.Random;

import org.kramerlab.timeseries.TimeSeries;
import org.kramerlab.timeseries.TimeSeriesDataset;

/**
 * @author razaa
 *
 */
public class RandomizedLegacyShapelets extends LegacyShapelets {

    private static Random rand;
    private static double percentage;
    
    public RandomizedLegacyShapelets(TimeSeriesDataset trainSet, int minLen,
                                     int maxLen, int stepSize) {
        super(trainSet, minLen, maxLen, stepSize);
        Properties props;
        try {
            props = new Properties();
            props.load(new FileInputStream("shapelets.properties"));
            percentage = Double.parseDouble(props.getProperty("selection_ratio", "50"));
            if (props.containsKey("rand_seed")) {
                int seed = Integer.parseInt(props.getProperty("rand_seed", "0"));
                rand = new Random(seed);
            } else {
                rand = new Random();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    protected TimeSeries getNextCandidate() {
        TimeSeries candidate = null;
        do {
            candidate = super.getNextCandidate();
        } while ( rand.nextFloat() > percentage/100 && (candidate != null) );
        return candidate;
    }
}
