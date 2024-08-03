package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class FeedForwardNetwork {
    private final int inputDim;
    private final int hiddenDim;
    private INDArray W1, W2, b1, b2;
    private INDArray hidden, activatedHidden;

    public FeedForwardNetwork(int inputDim, int hiddenDim) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        // Use Xavier initialization for weights
        this.W1 = Nd4j.randn(inputDim, hiddenDim).muli(Math.sqrt(2.0 / inputDim));
        this.W2 = Nd4j.randn(hiddenDim, inputDim).muli(Math.sqrt(2.0 / hiddenDim));
        this.b1 = Nd4j.zeros(1, hiddenDim);
        this.b2 = Nd4j.zeros(1, inputDim);
    }

    public INDArray forward(INDArray input) {
        hidden = input.mmul(W1).addRowVector(b1);
        activatedHidden = Transforms.relu(hidden); // ReLU activation
        return activatedHidden.mmul(W2).addRowVector(b2);
    }

    public void backward(INDArray input, INDArray gradOutput, double learningRate) {
        INDArray gradW2 = activatedHidden.transpose().mmul(gradOutput);
        INDArray gradb2 = gradOutput.sum(0);

        INDArray gradHidden = gradOutput.mmul(W2.transpose());

        INDArray reluGrad = hidden.gt(0);
        gradHidden = gradHidden.mul(reluGrad);

        INDArray gradW1 = input.transpose().mmul(gradHidden);
        INDArray gradb1 = gradHidden.sum(0);

        W1.subi(gradW1.mul(learningRate));
        W2.subi(gradW2.mul(learningRate));
        b1.subi(gradb1.mul(learningRate));
        b2.subi(gradb2.mul(learningRate));
    }
}
