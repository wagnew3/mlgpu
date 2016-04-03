package dynamicProgrammingValueIteration;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.State;

public class CoinFlipEnviroment extends Environment
{

	private CoinFlipState startState;
	private CoinFlipState currentState;
	
	private double headChance=0.0;
	
	public CoinFlipEnviroment(double headChance)
	{
		this.headChance=headChance;
	}
	
	@Override
	public State getCurrentState() 
	{
		return currentState;
	}

	@Override
	public void step(ActionPolicy policy) 
	{
		CoinFlipAction coinFlipAction=(CoinFlipAction)policy.getAction(currentState);
		if(Math.random()<headChance)
		{
			currentState=new CoinFlipState(currentState.money+coinFlipAction.betAmount);
		}
		else
		{
			currentState=new CoinFlipState(currentState.money-coinFlipAction.betAmount);
		}
	}

	@Override
	public void setStartState(State startState) 
	{
		this.startState=(CoinFlipState)startState;
		currentState=this.startState;
	}

	@Override
	public void reset() 
	{
		currentState=startState;
	}

	@Override
	public double transistionProbability(State oldState, State newState, Action action) 
	{
		CoinFlipState oldCoinFlipState=(CoinFlipState)oldState;
		CoinFlipState newCoinFlipState=(CoinFlipState)newState;
		CoinFlipAction coinFlipAction=(CoinFlipAction)action;
		
		if(oldCoinFlipState.money==0)
		{
			return 0.0;
		}
		
		if(oldCoinFlipState.money==100)
		{
			if(newCoinFlipState.money==0 && coinFlipAction.betAmount==0)
			{
				return 1.0;
			}
			else
			{
				return 0.0;
			}
		}
		
		if(oldCoinFlipState.money<coinFlipAction.betAmount)
		{
			return 0.0;
		}
		else if(oldCoinFlipState.money+coinFlipAction.betAmount==newCoinFlipState.money)
		{
			return headChance;
		}
		else if(oldCoinFlipState.money-coinFlipAction.betAmount==newCoinFlipState.money)
		{
			return 1.0-headChance;
		}
		else
		{
			return 0.0;
		}
	}

}
