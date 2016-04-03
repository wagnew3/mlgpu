package offPolicyMonteCarlo;

import ReinforcementMachineLearningFramework.Action;

public class CarRaceAction extends Action
{
	
	public int xAccel;
	public int yAccel;
	
	public CarRaceAction(int ID)
	{
		setActionFromID(ID);
	}
	
	public CarRaceAction(int xAccel, int yAccel)
	{
		this.xAccel=(int)(Math.signum(xAccel)*Math.min(Math.abs(xAccel), 1));
		this.yAccel=(int)(Math.signum(yAccel)*Math.min(Math.abs(yAccel), 1));
	}

	@Override
	public void setActionFromID(int ID) 
	{
		xAccel=ID%3-1;
		yAccel=(ID/3)%3-1;
	}

	@Override
	public int getActionID() 
	{
		return (xAccel+1)+3*(yAccel+1);
	}

}
