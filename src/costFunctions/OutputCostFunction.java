package costFunctions;

import org.apache.commons.math3.linear.ArrayRealVector;

public class OutputCostFunction extends CostFunction
{

	@Override
	public double getCost(ArrayRealVector input, ArrayRealVector networkOutput, ArrayRealVector desiredOutput) 
	{
		return desiredOutput.subtract(networkOutput).getL1Norm();
	}

	@Override
	public ArrayRealVector getCostDerivative(ArrayRealVector input, ArrayRealVector networkOutput, ArrayRealVector desiredOutput) 
	{
		ArrayRealVector derivative=new ArrayRealVector(networkOutput.getDimension());
		derivative.set(1.0);
		return derivative;
	}

}
