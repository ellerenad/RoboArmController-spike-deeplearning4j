package com.github.ellerenad.spike.deeplearning4j;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;

import java.io.File;
import java.io.IOException;

public class RoboArmInstructionClassifier {

    private static final int CLASSES_COUNT = 5;

    private MultiLayerNetwork model;
    private DataNormalization dataNormalizer;

    public RoboArmInstructionClassifier(String trainingSetName) throws Exception {
        String basePath = Constants.STORED_MODELS_PATH + trainingSetName + "/";
        model = loadModel(basePath);
        dataNormalizer = loadNormalizer(basePath);
    }

    /**
     * Predict a label given a domain object
     *
     * @param lineInput string containing the data of the position of the hands, separated by a pipe "|"
     * @return the predicted label
     */
    // This will be modified to transform the hand object on the RoboArmController project
    public int classify(String lineInput) {
        // Transform the data to the required format
        INDArray indArray = getArray(lineInput);

        // Normalize the data the same way it was normalized in the training phase
        dataNormalizer.transform(indArray);

        // Do the prediction
        INDArray result = model.output(indArray, false);

        // Get the index with the greatest probabilities
        int predictedLabel = getPredictedLabel(result);
        return predictedLabel;
    }

    private MultiLayerNetwork loadModel(String basePath) throws IOException {
        File locationToSaveModel = new File(basePath + Constants.STORED_MODEL_FILENAME);
        MultiLayerNetwork restoredModel = MultiLayerNetwork.load(locationToSaveModel, false);
        return restoredModel;
    }

    private DataNormalization loadNormalizer(String basePath) throws Exception {
        File locationToSaveNormalizer = new File(basePath + Constants.STORED_NORMALIZER_FILENAME);
        DataNormalization restoredNormalizer = NormalizerSerializer.getDefault().restore(locationToSaveNormalizer);
        return restoredNormalizer;
    }


    /**
     * Transform the data from the domain (in this case a string) to the object required by the framework to work
     *
     * @param line a line containing the data of the position of the hands, separated by a pipe "|"
     * @return an INDArray the framework can work with
     */
    // This will be modified to transform the hand object on the RoboArmController project
    private static INDArray getArray(String line) {
        String[] parts = line.split("\\|");
        float[] features = new float[16];
        for (int i = 0; i < 16; i++) {
            features[i] = Float.parseFloat(parts[i]);
        }

        NDArray ndArray = new NDArray(1, 16); // The empty constructor causes a NPE in add method
        DataBuffer dataBuffer = new FloatBuffer(features);
        ndArray.setData(dataBuffer);

        return ndArray;
    }

    /**
     * Get the index of the predicted label
     *
     * @param predictions INDArray with the probabilities per label
     * @return the index with the greatest probabilities
     */
    private static int getPredictedLabel(INDArray predictions) {
        int maxIndex = 0;
        // We should get max CLASSES_COUNT amount of predictions with probabilities.
        for (int i = 0; i < CLASSES_COUNT; i++) {
            if (predictions.getDouble(i) > predictions.getDouble(maxIndex)) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
