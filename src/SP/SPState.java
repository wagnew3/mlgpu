package SP;

import ReinforcementMachineLearningFramework.State;

public class SPState extends State
{
	
	int x; 
	int y;
	int endX;
	int endY;
	
	public SPState(int x, int y, int endX, int endY)
	{
		this.x=x;
		this.y=y;
		this.endX=endX;
		this.endY=endY;
	}

	@Override
	public int hashCode() 
	{
		return 100000*y+x;
	}

	@Override
	public boolean equals(Object other) 
	{
		if(!(other instanceof SPState))
		{
			return false;
		}
		else
		{
			return y==((SPState)other).y && x==((SPState)other).x;
		}
	}

	@Override
	public boolean isEndState() 
	{
		return x==endX && y==endY;
	}

}
