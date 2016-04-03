package ReinforcementMachineLearningFramework;

import java.io.Serializable;

public class ActionListElement implements Comparable<ActionListElement>, Serializable
{

	Action action;
	public double value;
	
	public ActionListElement(Action action, double value)
	{
		this.action=action;
		this.value=value;
	}
	
	public double getValue()
	{
		return value;
	}
	
	public void setValue(double value)
	{
		this.value=value;
	}
	
	public Action getAction()
	{
		return action;
	}
	
	@Override
	public int hashCode() 
	{
		return action.hashCode();
	}

	@Override
	public int compareTo(ActionListElement o) 
	{
		if(getAction().equals(o.getAction()))
		{
			return 0;
		}
		else if(getValue()>o.getValue())
		{
			return -1;
		}
		else if(getValue()<o.getValue())
		{
			return 1;
		}
		else
		{
			if(action.hashCode()==o.getAction().hashCode())
			{
				int u=0;
			}
			int a=action.hashCode();
			int b=o.getAction().hashCode();
			return action.hashCode()-o.getAction().hashCode();
		}
	}
	
	@Override
	public boolean equals(Object o)
	{
		if(o instanceof ActionListElement)
		{
			return getAction().equals(((ActionListElement)o).getAction());
		}
		else
		{
			return false;
		}
	}
	
}