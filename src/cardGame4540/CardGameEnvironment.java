package cardGame4540;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.State;


public class CardGameEnvironment extends Environment
{

	private CardGameState currentState;
	private CardGameState startState;
	
	@Override
	public State getCurrentState() 
	{
		return currentState;
	}

	@Override
	public int step(ActionPolicy policy) 
	{
		CardGamePolicy cardGamePolicy=(CardGamePolicy)policy;
		CardGameAction cardGameAction=(CardGameAction)cardGamePolicy.getAction(currentState);
		CardGameState newState=null;
		
		if(currentState.numberRedLeft+currentState.numberBlackLeft==2)
		{
			cardGameAction.action=1;	
		}
		if(Math.random()<((double)currentState.numberRedLeft)/((double)(currentState.numberRedLeft+currentState.numberBlackLeft)))
		{
			newState=new CardGameState(currentState.numberRedLeft-1, currentState.numberBlackLeft, cardGameAction, currentState);
		}
		else
		{
			newState=new CardGameState(currentState.numberRedLeft, currentState.numberBlackLeft-1, cardGameAction, currentState);
		}
		currentState=newState;
		if(cardGameAction.action==1)
		{
			return -1;	
		}
		else
		{
			return 0;
		}
	}

	@Override
	public void setStartState(State startState) 
	{
		this.startState=(CardGameState)startState;
		this.currentState=this.startState;
	}

	@Override
	public void reset() 
	{
		currentState=startState;
	}

	@Override
	public double transistionProbability(State oldState, State newState, Action action) 
	{
		CardGameState oldCardGameState=(CardGameState)oldState;
		CardGameState newCardGameState=(CardGameState)newState;
		if(oldCardGameState.numberRedLeft+oldCardGameState.numberBlackLeft==0)
		{
			return 0.0;
		}
		if(oldCardGameState.numberRedLeft-newCardGameState.numberRedLeft==1
				&& oldCardGameState.numberBlackLeft-newCardGameState.numberBlackLeft==0)
		{
			return ((double)oldCardGameState.numberRedLeft)/((double)(oldCardGameState.numberRedLeft+oldCardGameState.numberBlackLeft));
		}
		else if(oldCardGameState.numberRedLeft-newCardGameState.numberRedLeft==0
				&& oldCardGameState.numberBlackLeft-newCardGameState.numberBlackLeft==1)
		{
			return ((double)oldCardGameState.numberBlackLeft)/((double)(oldCardGameState.numberRedLeft+oldCardGameState.numberBlackLeft));
		}
		else
		{
			return 0.0;
		}
	}


}
