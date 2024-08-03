package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Transformer {
    private final TransformerLayer[] layers;

    public Transformer(int numLayers, int embeddingSize, int hiddenDim) {
        layers = new TransformerLayer[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layers[i] = new TransformerLayer(embeddingSize, hiddenDim);
        }
    }

    public INDArray forward(INDArray input) {
        INDArray output = input;
        for (TransformerLayer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    public void backward(INDArray input, INDArray gradOutput, double learningRate) {
        INDArray currentGradOutput = gradOutput;
        for (int i = layers.length - 1; i >= 0; i--) {
            layers[i].backward(input, currentGradOutput, learningRate);
        }
    }

    public TransformerLayer[] getLayers() {
        return layers;
    }
}
