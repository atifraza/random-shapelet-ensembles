package org.atif;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.StringTokenizer;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.kramerlab.timeseries.TimeSeries;

public class CommonConfig {
    // The following static String variables are the CLI switches
    protected String fileSw = "f";
    protected String fileSwL = "file";
    protected String methodSw = "m";
    protected String methodSwL = "method";
    protected String dataPath;
    protected String resultsPath;
    protected String paramsPath;
    protected CommandLine cmdLine;
    protected Options options;
    
    public CommonConfig(String[] args) {
        this.constructCommandLine(args);
        this.dataPath = System.getProperty("datapath", "../../data/UCR_2015");
        this.resultsPath = System.getProperty("resultspath", "./results/");
        this.paramsPath = "./params/";
    }
    
    public String getDataSetName() {
        return this.cmdLine.getOptionValue(fileSw, "");
    }
    
    public String getDataPath() {
        return this.dataPath;
    }
    
    public String getResultsPath() {
        return this.resultsPath;
    }
    
    public String getParamsPath() {
        return this.paramsPath;
    }
    
    public int getMethodType() {
        return Integer.parseInt(this.cmdLine.getOptionValue(methodSw, "1"));
    }
    
    protected void constructCommandLine(String[] args) {
        try {
            Option fName = new Option(fileSw, fileSwL, true, "dataset name");
            fName.setArgName("FILE");
            fName.setArgs(1);
            fName.setRequired(true);
            
            Option method = new Option(methodSw, methodSwL, true, "algorithm to use, 1: Legacy");
            method.setArgName("METHOD");
            method.setArgs(1);
            
            this.options = new Options();
            this.options.addOption(fName).addOption(method);
            
            CommandLineParser cliParser = new DefaultParser();
            
            this.cmdLine = cliParser.parse(options, args);
        } catch (ParseException e) {
            e.printStackTrace();
        }
    }
    
    public ArrayList<TimeSeries> loadDataset(Path pathToFile, String delimiter) {
        ArrayList<TimeSeries> dataset = null;
        TimeSeries currTS;
        ArrayList<Double> ts;
        int tsClass;
        
        String currLine;
        StringTokenizer st;
        try (BufferedReader br = Files.newBufferedReader(pathToFile)) {
            dataset = new ArrayList<TimeSeries>();
            while ((currLine = br.readLine()) != null) {
                if (currLine.matches("\\s*")) {
                    continue;
                } else {
                    currLine = currLine.trim();
                    st = new StringTokenizer(currLine, String.valueOf(delimiter));
                    tsClass = (int) Double.parseDouble(st.nextToken());
                    ts = new ArrayList<Double>();
                    while (st.hasMoreTokens()) {
                        ts.add(Double.parseDouble(st.nextToken()));
                    }
                    currTS = new TimeSeries(ts, tsClass);
                    dataset.add(currTS);
                }
            }
        } catch(Exception e) {
            e.printStackTrace();
        }
        return dataset;
    }
    
    /**
     * Print help to provided OutputStream.
     * 
     * @param options Command-line options to be part of usage.
     * @param detailed
     * @param out OutputStream to which to write the usage information.
     */
    public void printHelp(boolean detailed) {
        String cliSyntax = "java ";
        PrintWriter writer = new PrintWriter(System.out);
        HelpFormatter helpFormatter = new HelpFormatter();
        if (detailed) {
            helpFormatter.printHelp(writer, 120, cliSyntax, "", options, 7, 1, "", true);
        } else {
            helpFormatter.printUsage(writer, 120, cliSyntax, options);
        }
        writer.flush();
    }
}
