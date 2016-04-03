package dynamicProgrammingValueIteration;

import ReinforcementMachineLearningFramework.State;

public class CoinFlipState extends State
{

	int money;
	
	public CoinFlipState(int money)
	{
		this.money=money;
	}
	
	@Override
	public int getID() 
	{
		return money;
	}

	@Override
	public void createFromID(int ID) 
	{
		money=ID;
	}

}
