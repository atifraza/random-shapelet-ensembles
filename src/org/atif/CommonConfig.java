package org.atif;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Properties;
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
    protected String dataPathSwL = "data-path";
    protected String rsltPathSwL = "results-path";
    protected String paramsPathSwL = "params-path";
    protected CommandLine cmdLine;
    protected Options options;
    
    public CommonConfig(String[] args) {
        this.constructCommandLine(args);
    }
    
    public String getDataSetName() {
        return this.cmdLine.getOptionValue(fileSw, "");
    }
    
    public String getDataPath() {
        return this.cmdLine.getOptionValue(dataPathSwL, "../../data/UCR_2015");
    }
    
    public String getResultsPath() {
        return this.cmdLine.getOptionValue(rsltPathSwL, "./results");
    }
    
    public String getParamsPath() {
        return this.cmdLine.getOptionValue(paramsPathSwL, "./params");
    }
    
    protected void constructCommandLine(String[] args) {
        try {
            Option fName = Option.builder(fileSw)
                                 .longOpt(fileSwL)
                                 .argName("FILE")
                                 .required()
                                 .desc("Data set name")
                                 .numberOfArgs(1)
                                 .build();
            
            Option dataPath  = Option.builder("dp")
                                     .longOpt(dataPathSwL)
                                     .argName( "PATH-TO-DATA-FOLDER" )
                                     .desc("Specify the path to data file")
                                     .numberOfArgs(1)
                                     .build();
            
            Option resultsPath = Option.builder("rp")
                                       .longOpt(rsltPathSwL)
                                       .argName( "PATH-TO-RESULTS-FOLDER" )
                                       .desc("Specify the path to save results")
                                       .numberOfArgs(1)
                                       .build();
            
            Option paramsPath =  Option.builder("pp")
                                       .longOpt(paramsPathSwL)
                                       .argName( "PATH-TO-PARAMS-FOLDER" )
                                       .desc("Specify the path to param file")
                                       .numberOfArgs(1)
                                       .build();

            
            this.options = new Options();
            this.options.addOption(fName)
                        .addOption(dataPath)
                        .addOption(resultsPath)
                        .addOption(paramsPath);
            
            CommandLineParser cliParser = new DefaultParser();
            
            this.cmdLine = cliParser.parse(options, args);
        } catch (ParseException e) {
            System.err.println("Error parsing CommandLine Arguments\nProbably none specified");
        }
    }
    
    public Properties constructPropertiesObject(int tsLen) {
        Properties props = new Properties();
        try {
            File propsFile = new File(Paths.get(this.getParamsPath(), "default.params").toUri());
            props.load(new FileInputStream(propsFile));
            propsFile = new File(this.getParamsPath() + this.getDataSetName() + ".params");
            if (propsFile.exists()) {
                props.load(new FileInputStream(propsFile));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        if (!props.containsKey("minLen")) {
            props.setProperty("minLen", Integer.toString((int) Math.ceil(tsLen/4.0)));
        }
        if (!props.containsKey("maxLen")) {
            props.setProperty("maxLen", Integer.toString((int) Math.floor(tsLen*2/3.0)));
        }
        if (!props.containsKey("stepSize")) {
            props.setProperty("stepSize", "1");
        }
        
        return props;
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
