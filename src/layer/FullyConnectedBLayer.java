package layer;

import java.util.HashMap;

import org.apache.commons.math3.distribution.NormalDistribution;

import activationFunctions.ActivationFunction;
import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;

public class FullyConnectedBLayer extends BLayer
{
	
	protected int size;
	
	public FullyConnectedBLayer(ActivationFunction activationFunction, BLayer[] inputLayers) 
	{
		super(activationFunction, inputLayers);
	}
	
	public FullyConnectedBLayer(ActivationFunction activationFunction, BLayer[] inputLayers, int size) 
	{
		super(activationFunction, inputLayers);
		this.size=size;
		weights=new Matrix[inputLayers.length];
		
		double totalOutputSize=0.0;
		for(int intputLayerInd=0; intputLayerInd<inputLayers.length; intputLayerInd++)
		{
			totalOutputSize+=inputLayers[intputLayerInd].getOutputSize();
		}
		
		for(int intputLayerInd=0; intputLayerInd<inputLayers.length; intputLayerInd++)
		{
			weights[intputLayerInd]=new FDMatrix(new float[size][inputLayers[intputLayerInd].getOutputSize()]);
			NormalDistribution nInvGaussian=new NormalDistribution(0.0, 1.0/Math.sqrt(totalOutputSize));
			for(int rowIndex=0; rowIndex<weights[intputLayerInd].getRows(); rowIndex++)
			{
				for(int colIndex=0; colIndex<weights[intputLayerInd].getCols(); colIndex++)
				{
					weights[intputLayerInd].set(rowIndex, colIndex, (float)nInvGaussian.sample());
					//weights[intputLayerInd].set(rowIndex, colIndex, 0.01f);
				}
			}
		}
		
		biases=new FDMatrix(new float[size][1]);
		NormalDistribution zeroGuassian=new NormalDistribution(0.0, 1.0);
		for(int rowIndex=0; rowIndex<biases.getRows(); rowIndex++)
		{
			biases.set(rowIndex, 0, (float)zeroGuassian.sample());
			//biases.set(rowIndex, 0, 0.0f);
		}
		
		int u=0;
		//System.out.println("FullyConnectedBLayer fixed biases/weights");
	}

	@Override
	public Matrix getOutput(Matrix[] inputs, BLayer[] inputLayer, Matrix result) 
	{
		Matrix activation=result;
		activation.omatVecMultScaleAdd(weights[0], inputs[0], 1.0f, biases, activation, 0.0f);
		for(int inputInd=1; inputInd<inputs.length; inputInd++)
		{
			activation=activation.omatVecMultScaleAdd(weights[inputInd], inputs[inputInd], 1.0f, biases, activation, 1.0f);
		}
		
		return activationFunction.applyActivationFunction(activation);
	}

	@Override
	public Matrix getActivations(Matrix input, int weightIndex) 
	{
		Matrix activations=weights[weightIndex].matVecMultScaleAdd(weights[weightIndex], input, 1.0f, biases);
		return activations;
	}

	@Override
	public Matrix getOutputDerivatives(Matrix[] inputs, Matrix result) 
	{
		Matrix activation=result;
		activation.omatVecMultScaleAdd(weights[0], inputs[0], 1.0f, biases, activation, 0.0f);
		for(int inputInd=1; inputInd<inputs.length; inputInd++)
		{
			activation=activation.omatVecMultScaleAdd(weights[inputInd], inputs[inputInd], 1.0f, biases, activation, 1.0f);
		}
		
		return activationFunction.getDerivatives(activation);
	}
	
	public Matrix getConnectionsToPreviousLayer(BLayer prevLayer)
	{
		int prevLayerIndex=getPrevLayerIndex(prevLayer);
		float[][] connections=new float[weights[prevLayerIndex].getRows()][weights[prevLayerIndex].getCols()];
		
		for(int rowInd=0; rowInd<connections.length; rowInd++)
		{
			for(int colInd=0; colInd<connections[rowInd].length; colInd++)
			{
				connections[rowInd][colInd]=1.0f;
			}
		}
		
		return new FDMatrix(connections);
	}
	
	protected int getPrevLayerIndex(BLayer prevLayer)
	{
		int prevLayerIndex=0;
		for(; prevLayerIndex<inputLayers.length; prevLayerIndex++)
		{
			if(prevLayer.equals(inputLayers[prevLayerIndex]))
			{
				return prevLayerIndex;
			}
		}
		return -1;
	}

	@Override
	public Matrix getDeltas(Matrix[] nextLayerWeights, Matrix[] nextLayerDeltas, Matrix activationDerivative, Matrix weightsDeltasSum) 
	{
		for(int outputInd=0; outputInd<nextLayerWeights.length; outputInd++)
		{
			weightsDeltasSum=((FDMatrix)weightsDeltasSum).sgemv(true, 1.0f, nextLayerWeights[outputInd], 
					nextLayerDeltas[outputInd], 1, 1.0f, weightsDeltasSum, 1, true, weightsDeltasSum);
		}
		
		weightsDeltasSum=weightsDeltasSum.oebemult(activationDerivative);
		
		/*
		for(int rowInd=0; rowInd<weightsDeltasSum.getRows(); rowInd++)
		{
			weightsDeltasSum.set(rowInd, 0, weightsDeltasSum.get(rowInd, 0)*activationDerivative.get(rowInd, 0));
		}
		*/
		
		return weightsDeltasSum;
	}
	
	@Override
	public Matrix getWeightPDs(Matrix previousLayerOutputs, Matrix deltas, Matrix result) 
	{
		return deltas.outProd(deltas, previousLayerOutputs, result);
	}

	@Override
	public int getOutputSize() 
	{
		return size;
	}
	
}
