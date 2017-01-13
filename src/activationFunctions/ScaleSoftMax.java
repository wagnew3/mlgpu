package activationFunctions;

import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;

public class ScaleSoftMax extends ActivationFunction
{

	@Override
	public Matrix applyActivationFunction(Matrix input)
	{
		double expSum=0.0;
		double sum=0.0;
		for(int inputInd=0; inputInd<input.getLen(); inputInd++)
		{
			expSum+=Math.exp(Math.max(input.get(inputInd, 0), -5.0f));
			//sum+=input.get(inputInd, 0);
		}
		if(sum>0)
		{
			for(int inputInd=0; inputInd<input.getLen(); inputInd++)
			{
				input.set(inputInd, 0, (float)(Math.exp(input.get(inputInd, 0))/expSum));
				if(Float.isNaN((float)(Math.exp(input.get(inputInd, 0))/expSum)))
				{
					int u=0;
				}
			}
		}
		return input;
	}

	@Override
	public Matrix getDerivatives(Matrix input)
	{
		Matrix derivative=new FDMatrix(input.getLen(), 1);
		for(int inputInd=0; inputInd<input.getLen(); inputInd++)
		{
			for(int devInd=0; devInd<input.getLen(); devInd++)
			{
				if(inputInd==devInd)
				{
					derivative.set(devInd, 0,
							derivative.get(devInd, 0)
							+sigmoid(input.get(inputInd, 0))
							*(1-sigmoid(input.get(inputInd, 0))));
				}
				else
				{
					derivative.set(devInd, 0, 
							derivative.get(devInd, 0)
							+sigmoid(input.get(inputInd, 0))
							*-sigmoid(input.get(inputInd, 0)));
				}
			}
		}
		
		for(int devInd=0; devInd<input.getLen(); devInd++)
		{
			if(!Float.isFinite(derivative.get(devInd, 0)))
			{
				int u=0;
			}
		}
		
		return derivative;
	}
	
	protected float sigmoid(float activation)
	{
		return 1.0f/(1.0f+(float)Math.exp(-activation));
	}

}