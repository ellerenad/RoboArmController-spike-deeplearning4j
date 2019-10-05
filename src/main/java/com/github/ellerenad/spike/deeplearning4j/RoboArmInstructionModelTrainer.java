package com.github.ellerenad.spike.deeplearning4j;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

// Code somehow inspired on (though there are a lot of modifications) https://github.com/eugenp/tutorials/blob/master/deeplearning4j/src/main/java/com/baeldung/deeplearning4j/IrisClassifier.java
// Somehow this worked after i added the IteratorDataSetIterator  - apparently, having batches caused iterations to happen...
// actually, the difference was just using the iterator instead of the dataset on the fit method of the NN
// https://deeplearning4j.org/docs/latest/deeplearning4j-cheat-sheet#saving <- nice resource
// NOTES:
/*
Difficulties:
1) use an iterator instead of the dataset to train the Neural Network.
2) the data on the NDArray for prediction needs to be a float and not a double.
3) The empty constructor of the NDArray throws an NPE when we try to use the array.
4) the NN performs really poorly if we dont normalize it. If we do so, we need to also normalize the input for prediction
(obviously, right? it definitely didn't take 3 hours of debugging ;) ;)  Well, honestly this problem was combined with #2)
5) Might be kind of obvious, but still: The data needs to be shuffled, or the Neural Network will be a mess.
6) The Neural Network seems to be overfitting. It has a better performance with less data.


3 sessions to get this done. like 8h maybe? felt like: 25% getting the NN to train, 50% to get the prediction done, 25% all the other things (set up NN, prepare data...)
1 more session to refactor and clean
* */
public class RoboArmInstructionModelTrainer {

    private static final int CLASSES_COUNT = 5;
    private static final int LABEL_INDEX = 16;
    private static final int FEATURES_COUNT = 16;

    private static final double LEARNING_RATE = 0.1;
    private static final int BATCH_SIZE = 70;
    private static final int TOTAL_LINES = 5000;
    private static final double TRAIN_TO_TEST_RATIO = 0.6;
    private static final int SHUFFLE_SEED = 42;

    private final String TRAINING_SET_PATH;
    private final String BASE_PATH;

    public RoboArmInstructionModelTrainer(String trainingSetName) {
        TRAINING_SET_PATH = Constants.TRAINING_SETS_PATH + trainingSetName + Constants.TRAINING_SETS_FILE_EXTENSION;
        BASE_PATH = Constants.STORED_MODELS_PATH + trainingSetName + "/";
    }

    /**
     * Perform the whole training process, consisting in:
     * - Load the data
     * - Prepare it: shuffle, normalize
     * - Split into test and training sets
     * - Configure and train the Neural Network
     * - Store the model and the normalizer
     *
     * @return an object to evaluate the performance of the training of the Neural Network
     * @throws Exception
     */
    public Evaluation performTrainingProcess() throws Exception {

        // Load data
        DataSet allData = loadData(TRAINING_SET_PATH);

        // Prepare data
        allData.shuffle(SHUFFLE_SEED);

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);

        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(TRAIN_TO_TEST_RATIO); // The Neural Network seems to be overfitting - It has a better performance with less data
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        DataSetIterator trainingDataSetIterator = new IteratorDataSetIterator(trainingData.iterator(), BATCH_SIZE);

        // Configure Neural Network
        MultiLayerConfiguration configuration = getMultiLayerConfiguration();

        // Train Neural Network
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates
        model.fit(trainingDataSetIterator);
        //model.fit(trainingData); // This won't work. You kidding me? maybe because of the normalization?

        // Save the model and the normalizer
        store(model, normalizer, BASE_PATH);

        // Evaluate Neural Network
        return evaluate(model, testData);
    }

    private static void store(MultiLayerNetwork model, DataNormalization normalizer, String basePath) throws IOException {
        File locationToSaveModel = new File(basePath + Constants.STORED_MODEL_FILENAME);
        model.save(locationToSaveModel, false);

        File locationToSaveNormalizer = new File(basePath + Constants.STORED_NORMALIZER_FILENAME);
        NormalizerSerializer.getDefault().write(normalizer, locationToSaveNormalizer);
    }

    /**
     * Evaluate the trained MultiLayerNetwork
     *
     * @param testData the previously separated data to perform the test on
     * @param model    the model to test
     * @return Evaluation object
     */
    private static Evaluation evaluate(MultiLayerNetwork model, DataSet testData) {
        INDArray output = model.output(testData.getFeatures());
        Evaluation eval = new Evaluation(CLASSES_COUNT);
        eval.eval(testData.getLabels(), output);
        return eval;
    }

    /**
     * Load the data of a training set on a file
     *
     * @param path path relative to the classpath
     * @return a DataSet containing the data of the file
     * @throws IOException
     * @throws InterruptedException
     */
    private static DataSet loadData(String path) throws IOException, InterruptedException {
        DataSet allData;
        try (RecordReader recordReader = new CSVRecordReader(0, '|')) {
            recordReader.initialize(new FileSplit(new File(path)));
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, TOTAL_LINES, LABEL_INDEX, FEATURES_COUNT);
            allData = iterator.next();
        }
        return allData;
    }

    /**
     * Get the configuration of the Neural Network
     */
    private static MultiLayerConfiguration getMultiLayerConfiguration() {
        return new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .updater(new Nesterovs(LEARNING_RATE, 0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(3).build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(3).nOut(CLASSES_COUNT).build())
                .build();
    }

}



