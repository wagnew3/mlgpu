package layer;

import activationFunctions.ActivationFunction;
import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;

public class BInputLayer extends BLayer
{
	
protected int size;
	
	public BInputLayer(ActivationFunction activationFunction, BLayer[] inputLayers, int size) 
	{
		super(activationFunction, new BLayer[]{});
		this.size=size;
	}

	@Override
	public Matrix getOutput(Matrix[] input, BLayer[] inputLayer, Matrix result) 
	{
		return input[0];
	}

	@Override
	public Matrix getActivations(Matrix input, int weightIndex) 
	{
		System.out.println("Error: getting activation function output for input layer!");
		return null;
	}

	@Override
	public Matrix getOutputDerivatives(Matrix[] inputs, Matrix result) 
	{
		System.out.println("Error: getting output derivatives for input layer!");
		return null;
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
	public Matrix getDeltas(Matrix[] nextLayerWeights, Matrix[] nextLayerDeltas, Matrix activationDerivative) 
	{
		System.out.println("Error: getting deltas for input layer!");
		return null;
	}
	
	@Override
	public Matrix getWeightPDs(Matrix previousLayerOutputs, Matrix deltas, Matrix result) 
	{
		System.out.println("Error: getting weight PDs for input layer!");
		return null;
	}

	@Override
	public int getOutputSize() 
	{
		return size;
	}

}
