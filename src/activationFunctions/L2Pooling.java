package activationFunctions;

import org.apache.commons.math3.linear.ArrayRealVector;

public class L2Pooling extends PoolingActivationFunction
{

	@Override
	public double applyPoolingActivationFunction(ArrayRealVector input) 
	{
		return input.getNorm();
	}

	@Override
	public double getPoolingDerivatives(ArrayRealVector input) 
	{
		//derivative=(1/2)(1/(a^2+b^a+...)^0.5)(2a+2b+2c...)
		double derivative=0.0;
		for(int inputInd=0; inputInd<input.getDimension(); inputInd++)
		{
			derivative+=input.getEntry(inputInd);
		}
		derivative*=2;
		derivative*=0.5*(1/input.getLInfNorm());
		return derivative;
	}

}
