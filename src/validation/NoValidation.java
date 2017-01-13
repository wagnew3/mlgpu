package validation;

import nDimensionalMatrices.Matrix;
import network.SplitNetwork;

public class NoValidation extends Validator
{

	public NoValidation(Matrix[][] validationInputs, Matrix[][] validationOutputs) 
	{
		super(validationInputs, validationOutputs);
	}

	@Override
	public double validate(SplitNetwork network) 
	{
		return Double.NaN;
	}

}
