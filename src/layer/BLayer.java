package layer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import activationFunctions.ActivationFunction;
import nDimensionalMatrices.Matrix;

public abstract class BLayer implements Serializable 
{
	
	public ActivationFunction activationFunction;
	Matrix[] weights;
	public Matrix biases;
	public List<BLayer> outputLayers;
	BLayer[] inputLayers;
	
	public BLayer(ActivationFunction activationFunction, BLayer[] inputLayers)
	{
		this.activationFunction=activationFunction;
		outputLayers=new ArrayList<>();
		this.inputLayers=inputLayers;
		for(BLayer layer: inputLayers)
		{
			layer.outputLayers.add(this);
		}
	}
	
	public abstract Matrix getOutput(Matrix[] inputs, BLayer[] inputLayer, Matrix result);
	
	public abstract int getOutputSize();
	
	public abstract Matrix getActivations(Matrix inputs, int weightIndex);
	
	public abstract Matrix getOutputDerivatives(Matrix[] inputs, Matrix result);
	
	//weights.ebemult(connectionMatrix)^T * nextLayerDeltas.ebemult(activationDerivative)
	public abstract Matrix getDeltas(Matrix[] nextLayerWeights, Matrix nextLayerDeltas[], Matrix activationDerivative, Matrix weightsDeltasSum);
	
	public abstract Matrix getWeightPDs(Matrix previousLayerOutputs, Matrix deltas, Matrix result);
	
	public void updateWeights(Matrix[] weightPDs, float learningRate)
	{
		for(int weightsInd=0; weightsInd<weights.length; weightsInd++)
		{
			weights[weightsInd]=weights[weightsInd].omsubScale(weightPDs[weightsInd], learningRate);
		}
	}
	
	public void updateBiases(Matrix biasesPDs, float learningRate)
	{
		biases=biases.omsubScale(biasesPDs, learningRate);
	}
	
	public void setWeights(Matrix[] weights)
	{
		this.weights=weights;
	}
	
	public Matrix[] getWeights()
	{
		return weights;
	}
	
	public Matrix getBiases()
	{
		return biases;
	}
	
	public void setBiases(Matrix biases)
	{
		this.biases=biases;
	}
	
	public void setOutputLayers(BLayer[] outputLayers)
	{
		this.outputLayers=Arrays.asList(outputLayers);
	}
	
	public BLayer[] getOutputLayers()
	{
		return outputLayers.toArray(new BLayer[0]);
	}
	
	public BLayer[] getInputLayers()
	{
		return inputLayers;
	}
	
	public void setInputLayers(BLayer[] inputLayers)
	{
		this.inputLayers=inputLayers;
	}

}
