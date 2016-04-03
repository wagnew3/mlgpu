package SP;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;

public class SPEnvironment extends Environment
{

	int[][] region;
	int endX;
	int endY;
	SPState currentState;
	RewardFunction spRewardFunction;
	
	public SPEnvironment(int[][] region, int endX, int endY)
	{
		this.region=region;
		this.endX=endX;
		this.endY=endY;
		spRewardFunction=new SPRewardFunction();
	}
	
	@Override
	public void setStartState(State startState) 
	{
		currentState=(SPState)startState;
	}

	@Override
	public State getCurrentState() 
	{
		return currentState;
	}

	@Override
	public double takeAction(Action action) 
	{
		SPAction spAction=(SPAction)action;
		
		int newX=currentState.x+spAction.xDelta;
		if(newX<0)
		{
			newX=0;
		}
		else if(newX>=region.length)
		{
			newX=region.length-1;
		}
		int newY=currentState.y+spAction.yDelta;
		if(newY<0)
		{
			newY=0;
		}
		else if(newY>=region[0].length)
		{
			newY=region[0].length-1;
		}
		
		if(region[newX][newY]==1)
		{
			newX=currentState.x;
			newY=currentState.y;
		}
		
		SPState newState=new SPState(newX, newY, endX, endY);
		double reward=spRewardFunction.getReward(currentState, newState, spAction);
		currentState=newState;
		return reward;
	}

	@Override
	public Action getRandomAction(State state) 
	{
		return new SPAction((int)Math.round(2*Math.random()-1), (int)Math.round(2*Math.random()-1));
	}

}
