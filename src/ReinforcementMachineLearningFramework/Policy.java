package ReinforcementMachineLearningFramework;

public abstract class Policy 
{
	
	public abstract Action getAction(State state);
	
	public abstract void setAction(int state, int action);

	public abstract double stateActionChance(State state, Action action);
}
