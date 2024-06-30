package org.example;

import java.io.IOException;
import java.io.*;
import java.util.Arrays;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;

public class EmbeddingProcessor {
    private final WordVectors wordVectors;

    public EmbeddingProcessor(String modelPath) throws IOException {
        wordVectors = WordVectorSerializer.readWord2VecModel(new File(modelPath));
    }

    public double[] getWordEmbedding(String word) {
        return wordVectors.getWordVector(word);
    }

    public double[][] processText(String text) {
        String[] tokens = text.split("\\s+");
        int embeddingSize = getWordEmbedding(tokens[0]).length;
        double[][] embeddings = new double[tokens.length][embeddingSize];

        for (int i = 0; i < tokens.length; i++) {
            double[] wordEmbedding = getWordEmbedding(tokens[i]);
            double[] positionalEncoding = getPositionalEncoding(i, embeddingSize);
            embeddings[i] = addVectors(wordEmbedding, positionalEncoding);
        }

        return embeddings;
    }

    private double[] getPositionalEncoding(int position, int embeddingSize) {
        double[] encoding = new double[embeddingSize];
        for (int i = 0; i < embeddingSize; i++) {
            double angle = position / Math.pow(10000, (2 * (i / 2)) / (double) embeddingSize);
            encoding[i] = i % 2 == 0 ? Math.sin(angle) : Math.cos(angle);
        }
        return encoding;
    }

    private double[] addVectors(double[] vector1, double[] vector2) {
        double[] result = new double[vector1.length];
        for (int i = 0; i < vector1.length; i++) {
            result[i] = vector1[i] + vector2[i];
        }
        return result;
    }

    public static void main(String[] args) {
        try {
            EmbeddingProcessor processor = new EmbeddingProcessor("GoogleNews-vectors-negative300.bin");
            String inputText = "This is an example sentence for embeddings.";
            double[][] processedEmbeddings = processor.processText(inputText);

            for (double[] embedding : processedEmbeddings) {
                System.out.println(Arrays.toString(embedding));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
