package activationFunctions;

import java.io.Serializable;


import nDimensionalMatrices.*;

public abstract class ActivationFunction implements Serializable
{
	
	public ActivationFunction()
	{
		
	}
	
	public abstract Matrix applyActivationFunction(Matrix input);
	
	public abstract Matrix getDerivatives(Matrix input);
	
	/*
	public Matrix applyActivationFunction(Matrix input)
	{
		//applyActivationFunction(((FDMatrix)input));
		
		for(int inputIndex=0; inputIndex<input.getRows(); inputIndex++)
		{
			((FDMatrix)input).set(inputIndex, 0, applyActivationFunction(((FDMatrix)input).get(inputIndex, 0)));
			//((FDMatrix)input).data[inputIndex]=applyActivationFunction(((FDMatrix)input).data[inputIndex]);
		}
		
		return input;
	}

	
	public abstract float applyActivationFunction(float input);
	
	public Matrix getDerivatives(Matrix input)
	{
		Matrix derivatives=new FDMatrix(new float[input.getRows()], input.getRows(), input.getCols());
		for(int inputIndex=0; inputIndex<input.getRows(); inputIndex++)
		{
			((FDMatrix)derivatives).set(inputIndex, 0, getDerivative(((FDMatrix)input).get(inputIndex, 0)));
			//((FDMatrix)derivatives).data[inputIndex]=getDerivative(((FDMatrix)input).data[inputIndex]);
		}
		return derivatives;
	}
	
	public abstract float getDerivative(float input);
	*/
	
}
