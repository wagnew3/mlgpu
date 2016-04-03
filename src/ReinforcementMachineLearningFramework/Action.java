package ReinforcementMachineLearningFramework;

import java.io.Serializable;

public abstract class Action implements Serializable
{
	
	public abstract int hashCode();
	
	public abstract boolean equals(Object other);
	
	public abstract double[] getValue();
	
	public abstract Action getFromNNValue(double[] value);

}
