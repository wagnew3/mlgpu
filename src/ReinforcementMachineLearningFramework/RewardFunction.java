package ReinforcementMachineLearningFramework;

import java.io.Serializable;

public abstract class RewardFunction implements Serializable
{
	
	public abstract double getReward(State previousState, State currentState, Action action);

}
