package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SelfAttention {
    private final int embeddingSize;
    private final INDArray Wq, Wk, Wv;

    public SelfAttention(int embeddingSize) {
        this.embeddingSize = embeddingSize;

        this.Wq = Nd4j.rand(embeddingSize, embeddingSize);
        this.Wk = Nd4j.rand(embeddingSize, embeddingSize);
        this.Wv = Nd4j.rand(embeddingSize, embeddingSize);
    }

    public INDArray applySelfAttention(double[][] embeddings) {
        INDArray embeddingsMatrix = Nd4j.create(embeddings);

        INDArray Q = embeddingsMatrix.mmul(Wq);
        INDArray K = embeddingsMatrix.mmul(Wk);
        INDArray V = embeddingsMatrix.mmul(Wv);

        INDArray attentionScores = Q.mmul(K.transpose());

        double scale = Math.sqrt(embeddingSize);
        attentionScores = attentionScores.div(scale);

        INDArray attentionWeights = Nd4j.nn().softmax(attentionScores, 1);

        return attentionWeights.mmul(V);
    }

    public INDArray generateAnswer(INDArray contextVectors, INDArray questionVectors) {
        return null;
    }
}
