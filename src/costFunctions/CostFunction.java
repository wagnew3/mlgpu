package costFunctions;

import nDimensionalMatrices.Matrix;

public abstract class CostFunction 
{
	
	public CostFunction()
	{
		
	}

	public abstract float getCost(Matrix[] inputs, Matrix[] networkOutput, Matrix[] desiredOutput);
	
	public abstract Matrix[] getCostDerivative(Matrix[] inputs, Matrix[] networkOutput, Matrix[] desiredOutput, Matrix[] results);

}
