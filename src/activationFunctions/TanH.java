package activationFunctions;

public class TanH extends ActivationFunction
{
	
	public TanH()
	{
		
	}

	@Override
	public double applyActivationFunction(double input) 
	{
		return 2.0/(1+Math.exp(-2.0*input))-1.0;
	}

	@Override
	public double getDerivative(double input) 
	{

		return 1.0-Math.pow(applyActivationFunction(input), 2.0);
	}

}
