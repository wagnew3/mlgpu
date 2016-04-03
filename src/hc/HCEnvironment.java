package hc;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.LimitedActionsEnvironment;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;
import ReinforcementMachineLearningFramework.StateAction;

public class HCEnvironment extends LimitedActionsEnvironment
{

	int numberBits;
	int[] start;
	int numberMustBeZero;
	HCState currentState;
	RewardFunction hcRewardFunction;
	
	public HCEnvironment(int numberBits, int[] start, int numberMustBeZero)
	{
		this.numberBits=numberBits;
		this.start=start;
		this.numberMustBeZero=numberMustBeZero;
		hcRewardFunction=new HCRewardFunction(numberMustBeZero);
	}
	
	@Override
	public void setStartState(State startState) 
	{
		currentState=(HCState)startState;
	}

	@Override
	public State getCurrentState() 
	{
		return currentState;
	}

	@Override
	public double takeAction(Action action, int actor) 
	{
		HCAction spAction=(HCAction)action;
		
		int[] newPosition=new int[numberBits];
		System.arraycopy(currentState.position, 0, newPosition, 0, newPosition.length);
		
		for(int ind=0; ind<numberBits; ind++)
		{
			newPosition[ind]=(newPosition[ind]+spAction.move[ind])%2;
			if(spAction.move[ind]==1)
			{
				break;
			}
		}
		
		HCState newState=new HCState(newPosition, numberMustBeZero);
		double reward=hcRewardFunction.getReward(currentState, newState, spAction);
		currentState=newState;
		return reward;
	}

	@Override
	public Action getRandomAction(State state) 
	{
		int[] newPosition=new int[numberBits];
		int flipIndex=(int)(numberBits*Math.random());
		newPosition[flipIndex]=1;
		return new HCAction(newPosition);
	}

	@Override
	public StateAction[] getAllPossibleStateActions(State currentState) 
	{
		StateAction[] states=new StateAction[numberBits];
		for(int bitIndex=0; bitIndex<numberBits; bitIndex++)
		{
			int[] state=new int[numberBits];
			System.arraycopy(((HCState)currentState).getValueInt(), 0, state, 0, state.length);
			state[bitIndex]=(state[bitIndex]+1)%2;
			int[] action=new int[numberBits];
			action[bitIndex]=1;
			states[bitIndex]=new StateAction(new HCState(state, numberMustBeZero), new HCAction(action));
		}
		return states;
	}

}
