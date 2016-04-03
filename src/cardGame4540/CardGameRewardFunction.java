package cardGame4540;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;

public class CardGameRewardFunction extends RewardFunction
{
	
	public CardGameRewardFunction()
	{
		
	}

	@Override
	public double getReward(State previousState, State currentState, Action action) 
	{
		CardGameState cardGameState=(CardGameState)previousState;
		CardGameState currentCardGameState=(CardGameState)currentState;
		CardGameAction cardGameAction=(CardGameAction)action;
		
		if((cardGameState.numberRedLeft==currentCardGameState.numberRedLeft && cardGameState.numberBlackLeft==currentCardGameState.numberBlackLeft+1)
				|| (cardGameState.numberRedLeft==currentCardGameState.numberRedLeft+1 && cardGameState.numberBlackLeft==currentCardGameState.numberBlackLeft))
		{
			if(cardGameAction==null || cardGameAction.action==0)
			{
				return 0.0;
			}
			else
			{
				if(Math.random()<(double)cardGameState.numberRedLeft/((double)(cardGameState.numberRedLeft+cardGameState.numberBlackLeft)))
				{
					return 1.0;
				}
				else
				{
					return -1.0;
				}
			}
		}
		else
		{
			return 0.0;
		}
	}

}
