package dynamicProgrammingPolicyIteration;

import ReinforcementMachineLearningFramework.Action;

public class CarRentalAction extends Action 
{
	
	private int numberCarsToMove;
	
	public CarRentalAction(int numberCarsToMove)
	{
		this.numberCarsToMove=numberCarsToMove;
	}
	
	public int getNumberCarsToMove()
	{
		return numberCarsToMove;
	}

	@Override
	public void setActionFromID(int ID) 
	{
		numberCarsToMove=ID-5;
	}

	@Override
	public int getActionID() 
	{
		return numberCarsToMove+5;
	}

}
