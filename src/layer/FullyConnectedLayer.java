package layer;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import activationFunctions.ActivationFunction;

public class FullyConnectedLayer extends Layer
{

	protected int size;
	
	public FullyConnectedLayer(ActivationFunction activationFunction, Layer previousLayer, int size) 
	{
		super(activationFunction);
		this.size=size;
		
		weights=new BlockRealMatrix(size, previousLayer.getOutputSize());
		NormalDistribution nInvGaussian=new NormalDistribution(0.0, 1.0/Math.sqrt(previousLayer.getOutputSize()));
		for(int rowIndex=0; rowIndex<weights.getRowDimension(); rowIndex++)
		{
			for(int colIndex=0; colIndex<weights.getColumnDimension(); colIndex++)
			{
				weights.setEntry(rowIndex, colIndex, nInvGaussian.sample());
			}
		}
		
		biases=new ArrayRealVector(size);
		NormalDistribution zeroGuassian=new NormalDistribution(0.0, 1.0);
		for(int rowIndex=0; rowIndex<biases.getDimension(); rowIndex++)
		{
			biases.setEntry(rowIndex, zeroGuassian.sample());
		}
	}

	public FullyConnectedLayer(ActivationFunction activationFunction) 
	{
		super(activationFunction);
	}

	@Override
	public ArrayRealVector getOutput(ArrayRealVector input) 
	{
		ArrayRealVector activations=getActivations(input);
		ArrayRealVector output=activationFunction.applyActivationFunction(activations);
		return output;
	}
	
	@Override
	public ArrayRealVector getActivations(ArrayRealVector input) 
	{
		ArrayRealVector activations=(ArrayRealVector)weights.operate(input).add(biases);
		return activations;
	}

	@Override
	public ArrayRealVector getOutputDerivatives(ArrayRealVector input) 
	{
		ArrayRealVector activations=getActivations(input);
		ArrayRealVector outputDerivatives=activationFunction.getDerivatives(activations);
		return outputDerivatives;
	}
	
	public BlockRealMatrix getConnectionsToPreviousLayer()
	{
		return new BlockRealMatrix(weights.getRowDimension(), weights.getColumnDimension()).scalarAdd(1.0);
	}

	@Override
	public ArrayRealVector getDeltas(BlockRealMatrix nextLayerWeights, ArrayRealVector nextLayerDeltas, ArrayRealVector activationDerivative) 
	{
		return (ArrayRealVector)nextLayerWeights.transpose().operate(nextLayerDeltas).ebeMultiply(activationDerivative);
	}
	
	@Override
	public RealMatrix getWeightPDs(ArrayRealVector previousLayerOutputs, ArrayRealVector deltas) 
	{
		return deltas.outerProduct(previousLayerOutputs);
	}

	@Override
	public int getOutputSize() 
	{
		return size;
	}
	
	public Layer clone()
	{
		BlockRealMatrix newWeights=weights.copy();
		ArrayRealVector newBiases=biases.copy();
		FullyConnectedLayer layer=new FullyConnectedLayer(activationFunction);
		layer.weights=newWeights;
		layer.biases=newBiases;
		layer.size=size;
		return layer;
	}

}
