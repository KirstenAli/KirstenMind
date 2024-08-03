package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class LayerNorm {
    private final int embeddingSize;
    private final INDArray gamma;
    private final INDArray beta;

    public LayerNorm(int embeddingSize) {
        this.embeddingSize = embeddingSize;
        this.gamma = Nd4j.ones(embeddingSize);
        this.beta = Nd4j.zeros(embeddingSize);
    }

    public INDArray normalize(INDArray input) {
        INDArray mean = input.mean(1);
        INDArray variance = input.var(1);
        INDArray normalized = input.subColumnVector(mean).divColumnVector(Transforms.sqrt(variance.add(1e-5)));
        return normalized.mulRowVector(gamma).addRowVector(beta);
    }
}
