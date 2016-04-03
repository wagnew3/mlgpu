package SP;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;

public class SPRewardFunction extends RewardFunction
{

	double movePenalty=0.001;
	double endStateReward=1.0;
	
	public SPRewardFunction()
	{
	}
	
	@Override
	public double getReward(State previousState, State currentState, Action action) 
	{
		if(currentState.isEndState())
		{
			return endStateReward;
		}
		else
		{
			return -movePenalty;
		}
	}

}
