package ReinforcementMachineLearningFramework;

import java.io.Serializable;

public abstract class CompletelyKnownEnvironment implements Serializable 
{
	
	public abstract State getCurrentState();
	
	/*
	 * -1=end state reached
	 */
	public abstract int step(Policy policy);
	
	public abstract void setStartState(State startState);
	
	public abstract void reset();
	
	public abstract double transistionProbability(State oldState, State newState, Action action);

}
