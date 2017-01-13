package activationFunctions;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import jcuda.Pointer;
import jcuda.Sizeof;
import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;

public class RectifiedLinearActivationFunction extends ActivationFunction
{

	/*
	@Override
	public float applyActivationFunction(float input) 
	{
		if(input>=0)
		{
			return input;
		}
		else
		{
			return 0.01f*input;
		}
	}

	@Override
	public float getDerivative(float input) 
	{
		if(input>=0)
		{
			return 1.0f;
		}
		else
		{
			return 0.01f;
		}
	}
	*/
	
	float min=-10;
	float max=100;
	
	@Override
	public Matrix applyActivationFunction(Matrix input)
	{
		for(int inputInd=0; inputInd<input.getLen(); inputInd++)
		{
			if(input.get(inputInd, 0)<0)
			{
				input.set(inputInd, 0, 0.01f*input.get(inputInd, 0));
			}
			else
			{
				
			}
		}
        return input;
	}
	
	@Override
	public Matrix getDerivatives(Matrix input)
	{
		for(int inputInd=0; inputInd<input.getLen(); inputInd++)
		{
			if(input.get(inputInd, 0)<0)
			{
				input.set(inputInd, 0, 0.01f);
			}
			else
			{
				input.set(inputInd, 0, 1.0f);
			}
		}
        return input;
	}

}
