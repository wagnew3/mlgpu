package activationFunctions;

public class RectifiedLinearActivationFunction extends ActivationFunction
{

	@Override
	public double applyActivationFunction(double input) 
	{
		if(input>=0)
		{
			return Math.log(input);
		}
		else
		{
			return 0.5*input;
		}
	}

	@Override
	public double getDerivative(double input) 
	{
		if(input>=0)
		{
			return 1/input;
		}
		else
		{
			return 0.5;
		}
	}

}
