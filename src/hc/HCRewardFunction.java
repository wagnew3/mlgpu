package hc;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.RewardFunctionWithHeuristic;
import ReinforcementMachineLearningFramework.State;

public class HCRewardFunction extends RewardFunctionWithHeuristic
{

	double movePenalty=0.1;
	double endStateReward=0.5;
	protected int numberZero;
	
	public HCRewardFunction(int numberZero)
	{
		this.numberZero=numberZero;
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

	@Override
	public double getHeuristicReward(State previousState, State currentState, Action action) 
	{
		int numPrevStateZeros=0;
		for(int ind=0; ind<numberZero; ind++)
		{
			if(previousState.getValue()[ind]==0.0)
			{
				numPrevStateZeros++;
			}
		}
		
		int numCurrentStateZeros=0;
		for(int ind=0; ind<numberZero; ind++)
		{
			if(currentState.getValue()[ind]==0.0)
			{
				numCurrentStateZeros++;
			}
		}
		return numCurrentStateZeros-numPrevStateZeros;
	}

}
