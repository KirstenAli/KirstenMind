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

    public INDArray applySelfAttention(INDArray embeddingsMatrix) {
        INDArray Q = embeddingsMatrix.mmul(Wq);
        INDArray K = embeddingsMatrix.mmul(Wk);
        INDArray V = embeddingsMatrix.mmul(Wv);

        INDArray attentionScores = Q.mmul(K.transpose());

        double scale = Math.sqrt(embeddingSize);
        attentionScores = attentionScores.div(scale);

        INDArray attentionWeights = Nd4j.nn().softmax(attentionScores, 1);

        return attentionWeights.mmul(V);
    }

    public void backward(INDArray input, INDArray gradOutput, double learningRate) {
        INDArray Q = input.mmul(Wq);
        INDArray K = input.mmul(Wk);
        INDArray V = input.mmul(Wv);

        INDArray attentionScores = Q.mmul(K.transpose());
        double scale = Math.sqrt(embeddingSize);
        attentionScores = attentionScores.div(scale);

        INDArray attentionWeights = Nd4j.nn().softmax(attentionScores, 1);

        INDArray gradAttentionWeights = gradOutput.mmul(V.transpose());
        INDArray gradV = attentionWeights.transpose().mmul(gradOutput);

        INDArray gradQ = gradAttentionWeights.mmul(K);
        INDArray gradK = gradAttentionWeights.transpose().mmul(Q);

        INDArray gradWq = input.transpose().mmul(gradQ);
        INDArray gradWk = input.transpose().mmul(gradK);
        INDArray gradWv = input.transpose().mmul(gradV);

        Wq.subi(gradWq.mul(learningRate));
        Wk.subi(gradWk.mul(learningRate));
        Wv.subi(gradWv.mul(learningRate));
    }
}
