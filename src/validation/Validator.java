package validation;

import nDimensionalMatrices.Matrix;
import network.SplitNetwork;

public abstract class Validator 
{
	
	Matrix[][] validationInputs;
	Matrix[][] validationOutputs;
	
	public Validator(Matrix[][] validationInputs, Matrix[][] validationOutputs)
	{
		this.validationInputs=validationInputs;
		this.validationOutputs=validationOutputs;
	}
	
	public abstract double validate(SplitNetwork network);
}
