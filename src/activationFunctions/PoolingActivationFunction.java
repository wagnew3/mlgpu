package activationFunctions;

import java.io.Serializable;

import org.apache.commons.math3.linear.ArrayRealVector;

public abstract class PoolingActivationFunction extends ActivationFunction implements Serializable
{

	@Override
	public double applyActivationFunction(double input) 
	{
		System.out.println("Calling applyActivationFunction on a PoolingActivationFunction!");
		return 0;
	}

	@Override
	public double getDerivative(double input) 
	{
		System.out.println("Calling getDerivative on a PoolingActivationFunction!");
		return 0;
	}

}
