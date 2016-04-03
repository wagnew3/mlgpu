package ReinforcementMachineLearningFramework;

import java.io.Serializable;

public abstract class State implements Serializable
{

	public abstract int hashCode();
	
	public abstract boolean equals(Object other);
	
	public abstract boolean isEndState();
	
	public abstract double[] getValue();
	
	public abstract State getFromNNValue(double[] value);
	
	public abstract double[] getNNValue();

}
