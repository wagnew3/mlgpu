package dynamicProgrammingValueIteration;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.State;

public class CoinFlipPolicy extends ActionPolicy
{

	int[] policy;
	
	public CoinFlipPolicy(int numberStates)
	{
		policy=new int[numberStates];
	}
	
	@Override
	public Action getAction(State state) 
	{
		return new CoinFlipAction(policy[((CoinFlipState)state).money]);
	}

	@Override
	public void setAction(int state, int action) 
	{
		policy[state]=Math.min(action, state);
	}

}
