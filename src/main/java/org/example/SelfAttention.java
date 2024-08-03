package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SelfAttention {
    private final int embeddingSize;
    private INDArray Wq, Wk, Wv;
    private INDArray Q, K, V, attentionWeights, attentionScores;

    public SelfAttention(int embeddingSize) {
        this.embeddingSize = embeddingSize;
        this.Wq = Nd4j.rand(embeddingSize, embeddingSize);
        this.Wk = Nd4j.rand(embeddingSize, embeddingSize);
        this.Wv = Nd4j.rand(embeddingSize, embeddingSize);
    }

    public INDArray applySelfAttention(INDArray embeddingsMatrix) {
        Q = embeddingsMatrix.mmul(Wq);
        K = embeddingsMatrix.mmul(Wk);
        V = embeddingsMatrix.mmul(Wv);

        attentionScores = Q.mmul(K.transpose());

        double scale = Math.sqrt(embeddingSize);
        attentionScores = attentionScores.div(scale);

        attentionWeights = Nd4j.nn().softmax(attentionScores, 1);

        return attentionWeights.mmul(V);
    }

    public void backward(INDArray input, INDArray gradOutput, double learningRate) {
        INDArray gradV = attentionWeights.transpose().mmul(gradOutput);

        INDArray gradAttentionWeights = gradOutput.mmul(V.transpose());

        INDArray gradAttentionScores = gradAttentionWeights.mul(attentionWeights)
                .sub(attentionWeights.mul(attentionWeights.mul(gradAttentionWeights).sum(1).reshape(1, -1)));

        gradAttentionScores = gradAttentionScores.div(Math.sqrt(embeddingSize));

        INDArray gradQ = gradAttentionScores.mmul(K);
        INDArray gradK = gradAttentionScores.transpose().mmul(Q);

        INDArray gradWq = input.transpose().mmul(gradQ);
        INDArray gradWk = input.transpose().mmul(gradK);
        INDArray gradWv = input.transpose().mmul(gradV);

        Wq.subi(gradWq.mul(learningRate));
        Wk.subi(gradWk.mul(learningRate));
        Wv.subi(gradWv.mul(learningRate));
    }
}
