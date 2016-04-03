package activationFunctions;

import org.apache.commons.math3.linear.ArrayRealVector;

public class SoftMax extends PoolingActivationFunction
{

	@Override
	public ArrayRealVector applyActivationFunction(ArrayRealVector input)
	{
		double expSum=0.0;
		for(int inputInd=0; inputInd<input.getDimension(); inputInd++)
		{
			expSum+=Math.exp(input.getEntry(inputInd));
		}
		ArrayRealVector result=new ArrayRealVector(input.getDimension());
		for(int inputInd=0; inputInd<input.getDimension(); inputInd++)
		{
			result.setEntry(inputInd, Math.exp(input.getEntry(inputInd))/expSum);
		}
		for(int inputInd=0; inputInd<input.getDimension(); inputInd++)
		{
			if(Double.isNaN(result.getEntry(inputInd)))
			{
				int u=0;
			}
		}
		return result;
	}

	@Override
	public ArrayRealVector getDerivatives(ArrayRealVector input)
	{
		Sigmoid sigmoid=new Sigmoid();
		ArrayRealVector derivative=new ArrayRealVector(input.getDimension());
		for(int inputInd=0; inputInd<input.getDimension(); inputInd++)
		{
			for(int devInd=0; devInd<input.getDimension(); devInd++)
			{
				if(inputInd==devInd)
				{
					derivative.setEntry(devInd, 
							derivative.getEntry(devInd)
							+sigmoid.applyActivationFunction(input.getEntry(inputInd))
							*(1-sigmoid.applyActivationFunction(input.getEntry(inputInd))));
				}
				else
				{
					derivative.setEntry(devInd, 
							derivative.getEntry(devInd)
							+sigmoid.applyActivationFunction(input.getEntry(inputInd))
							*-sigmoid.applyActivationFunction(input.getEntry(inputInd)));
				}
			}
		}
		return derivative;
	}

}
