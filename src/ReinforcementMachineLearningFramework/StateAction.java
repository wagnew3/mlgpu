package ReinforcementMachineLearningFramework;

import java.io.Serializable;

public class StateAction implements Serializable 
{
	
	public State state;
	public Action action;
	StateAction previousStateAction;
	public double eligibility;
	
	public StateAction(State state, Action action)
	{
		this.state=state;
		this.action=action;
		this.previousStateAction=previousStateAction;
	}
	
	public State getState()
	{
		return state;
	}
	
	public Action getAction()
	{
		return action;
	}
	
	public double getEligibility()
	{
		return eligibility;
	}
	
	public void setEligibility(double newEligibility)
	{
		eligibility=newEligibility;
	}
	
	public StateAction getPreviousStateAction()
	{
		return previousStateAction;
	}
	
	public void setPreviousStateAction(StateAction newPreviousStateAction)
	{
		previousStateAction=newPreviousStateAction;
	}
	
	public int hashCode()
	{
		return state.hashCode()*action.hashCode();
	}
	
	public boolean equals(Object other)
	{
		if(!(other instanceof StateAction))
		{
			return false;
		}
		else
		{
			return ((StateAction)other).getState().equals(getState()) && ((StateAction)other).getAction().equals(getAction());
		}
	}
	
}
