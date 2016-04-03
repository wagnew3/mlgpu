package layer;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;

import activationFunctions.ActivationFunction;

public class InputLayer extends Layer
{

	protected int size;
	
	public InputLayer(ActivationFunction activationFunction, int size) 
	{
		super(activationFunction);
		this.size=size;
	}

	@Override
	public ArrayRealVector getOutput(ArrayRealVector input) 
	{
		return input;
	}
	
	public ArrayRealVector getActivationFuncOutput(ArrayRealVector activations)
	{
		System.out.println("Error: getting activation function output for input layer!");
		return null;
	}

	@Override
	public int getOutputSize() 
	{
		return size;
	}

	@Override
	public ArrayRealVector getActivations(ArrayRealVector input) 
	{
		return input;
	}

	@Override
	public ArrayRealVector getOutputDerivatives(ArrayRealVector output)
	{
		System.out.println("Error: getting output derivatives for input layer!");
		return null;
	}

	@Override
	public ArrayRealVector getDeltas(BlockRealMatrix nextLayerWeights,
			ArrayRealVector nextLayerDeltas, ArrayRealVector activationDerivative) 
	{
		System.out.println("Error: getting deltas for input layer!");
		return null;
	}
	
	@Override
	public BlockRealMatrix getWeightPDs(ArrayRealVector previousLayerOutputs, ArrayRealVector deltas) 
	{
		System.out.println("Error: getting weight PDs for input layer!");
		return null;
	}
	
	public Layer clone()
	{
		InputLayer layer=new InputLayer(activationFunction, size);
		return layer;
	}

}
