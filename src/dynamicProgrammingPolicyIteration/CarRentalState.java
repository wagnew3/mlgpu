package dynamicProgrammingPolicyIteration;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.State;

public class CarRentalState extends State
{
	
	public int numberCarsLocation1;
	public int numberCarsLocation2;
	
	public int numberCarsReturnedLocation1;
	public int numberCarsReturnedLocation2;
	
	public CarRentalState predecessor=null;
	
	public CarRentalState(int numberCarsLocation1, int numberCarsLocation2, CarRentalState predecessor)
	{
		this.numberCarsLocation1=numberCarsLocation1;
		this.numberCarsLocation2=numberCarsLocation2;
		this.predecessor=predecessor;
	}

	@Override
	public State getSuccessor(Action action) 
	{
		CarRentalAction carRentalAction=(CarRentalAction)action;
		if(carRentalAction.getNumberCarsToMove()>0)
		{
			carRentalAction.setActionFromID(Math.min(carRentalAction.getNumberCarsToMove(), numberCarsLocation1));
		}
		else
		{
			carRentalAction.setActionFromID(Math.min(-carRentalAction.getNumberCarsToMove(), numberCarsLocation1));
		}
		return new CarRentalState(Math.min(numberCarsLocation1-carRentalAction.getNumberCarsToMove(), 20),
				Math.min(numberCarsLocation2+carRentalAction.getNumberCarsToMove(), 20),
				this);
	}

	@Override
	public int getID()
	{
		return 21*numberCarsLocation1+numberCarsLocation2;
	}

	@Override
	public void createFromID(int ID) 
	{
		numberCarsLocation1=ID/21;
		numberCarsLocation2=ID%21;
	}

}
