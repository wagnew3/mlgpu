package dynamicProgrammingValueIteration;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;

public class CoinFlipReward extends RewardFunction
{

	public CoinFlipReward()
	{
		
	}
	
	@Override
	public double getReward(State previousState, State currentState, Action action)
	{
		if(((CoinFlipState)previousState).money!=100 && ((CoinFlipState)currentState).money==100)
		{
			return 1.0;
		}
		else
		{
			return 0.0;
		}
	}

}
