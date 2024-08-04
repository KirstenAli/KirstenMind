package org.example;

import org.nd4j.linalg.api.ndarray.INDArray;

public class TransformerLayer {
    private final SelfAttention selfAttention;
    private final FeedForwardNetwork ffn;
    private final LayerNorm layerNorm1;
    private final LayerNorm layerNorm2;

    public TransformerLayer(int embeddingSize, int hiddenDim) {
        selfAttention = new SelfAttention(embeddingSize);
        ffn = new FeedForwardNetwork(embeddingSize, hiddenDim);
        layerNorm1 = new LayerNorm(embeddingSize);
        layerNorm2 = new LayerNorm(embeddingSize);
    }

    public INDArray forward(INDArray input) {
        // Self-Attention sublayer
        INDArray attentionOutput = selfAttention.applySelfAttention(input);
        INDArray attentionNorm = layerNorm1.normalize(input.add(attentionOutput)); // Add & Norm

        // Feed-Forward Network sublayer
        INDArray ffnOutput = ffn.forward(attentionNorm);
        return layerNorm2.normalize(attentionNorm.add(ffnOutput)); // Add & Norm
    }

    public void backward(INDArray input, INDArray gradOutput, double learningRate) {
        //TODO
    }

    public SelfAttention getSelfAttention() {
        return selfAttention;
    }

    public FeedForwardNetwork getFeedForwardNetwork() {
        return ffn;
    }
}
