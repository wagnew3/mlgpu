package SP;

import ReinforcementMachineLearningFramework.Action;

public class SPAction extends Action
{

	int xDelta;
	int yDelta;
	
	public SPAction(int xDelta, int yDelta)
	{
		this.xDelta=xDelta;
		this.yDelta=yDelta;
	}
	
	@Override
	public int hashCode() 
	{
		return 10*yDelta+xDelta;
	}

	@Override
	public boolean equals(Object other) 
	{
		if(!(other instanceof SPAction))
		{
			return false;
		}
		else
		{
			return yDelta==((SPAction)other).yDelta && xDelta==((SPAction)other).xDelta;
		}
	}

}
