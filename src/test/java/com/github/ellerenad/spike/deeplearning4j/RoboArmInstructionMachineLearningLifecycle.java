package com.github.ellerenad.spike.deeplearning4j;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.Test;
import org.nd4j.evaluation.classification.Evaluation;

import java.io.File;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RoboArmInstructionMachineLearningLifecycle {
    private Logger log = Logger.getLogger(RoboArmInstructionMachineLearningLifecycle.class.getName());

    @Test
    void testLifecycle() throws Exception {
        try {
            String trainingSetName = "trainingSet_1559827006805";

            // Test training and evaluate model
            RoboArmInstructionModelTrainer roboArmInstructionModelTrainer = new RoboArmInstructionModelTrainer(trainingSetName);

            Evaluation modelEvaluation = roboArmInstructionModelTrainer.performTrainingProcess();
            log.info(modelEvaluation.stats());
            // Arbitrary values -> good enough to work with them
            assertTrue(modelEvaluation.accuracy() > 0.9);
            assertTrue(modelEvaluation.precision() > 0.9);

            // Test loading and perform some predictions
            RoboArmInstructionClassifier roboArmInstructionClassifier = new RoboArmInstructionClassifier(trainingSetName);

            // Note 1: Data directly taken from the data file. It might be the case that this data was used to train
            // the Neural Network - Testing the NN with data it was trained on should be avoided.
            // Note 2: The Neural Network has an accuracy of around 90% - it might be the case these labels are not
            // properly predicted, and breaks the test, though the value is still accepted. This could break a CI/CD pipeline. As this is a MVP, this is accepted.
            assertEquals(0, roboArmInstructionClassifier.classify("0|38.208|156.451|80.1864|12.8463|168.467|-35.969|-27.7574|170.312|-61.8114|-58.2967|168.344|-58.3535|-108.05|159.967|-39.629")); // expected 0
            assertEquals(1, roboArmInstructionClassifier.classify("0|42.3547|159.677|54.8158|-5.02637|182.497|-62.1005|-110.879|167.142|-57.1002|-45.0468|94.0677|-25.2277|-64.9443|94.9761|-12.2357")); // expected 1
            assertEquals(2, roboArmInstructionClassifier.classify("0|-23.8724|119.959|6.16023|-26.6377|128.075|-2.1761|-41.7599|121.839|-0.334683|-63.474|122.708|1.38042|-78.5949|131.032|4.72392")); // expected 2
            assertEquals(3, roboArmInstructionClassifier.classify("1|-57.9509|154.695|75.8706|-48.7798|153.478|-43.7015|-11.3466|152.738|-74.2601|37.5707|144.084|-78.3605|85.2095|127.172|-55.4767")); // expected 3
            assertEquals(4, roboArmInstructionClassifier.classify("1|-57.7108|164.532|57.0743|9.09145|65.1718|26.2156|16.9844|60.849|30.1358|29.3373|65.982|25.5115|35.3826|78.686|13.2991")); // expected 4
        } finally {
            // cleanup
            File outputDirectory = new File(Constants.STORED_MODELS_PATH);
            // comment out the lines below out if you want to keep the trained model
            FileUtils.deleteDirectory(outputDirectory);
        }

    }

}