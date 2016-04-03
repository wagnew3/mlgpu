package dynamicProgrammingValueIteration;

import ReinforcementMachineLearningFramework.Action;

public class CoinFlipAction extends Action
{
	
	public int betAmount;
	
	public CoinFlipAction(int betAmount)
	{
		this.betAmount=betAmount;
	}

	@Override
	public void setActionFromID(int ID) 
	{
		betAmount=ID;
	}

	@Override
	public int getActionID() 
	{
		return betAmount;
	}

}
