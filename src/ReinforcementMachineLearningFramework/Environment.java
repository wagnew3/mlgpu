package ReinforcementMachineLearningFramework;

import java.io.Serializable;

public abstract class Environment implements Serializable
{
	
	public abstract void setStartState(State startState);
	
	public abstract State getCurrentState();
	
	public abstract double takeAction(Action action, int actor);
	
	public abstract StateAction getRandomStateAction(State state);
	
	public abstract Environment clone();
	
}
