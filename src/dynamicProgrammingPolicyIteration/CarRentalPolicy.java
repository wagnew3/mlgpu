package dynamicProgrammingPolicyIteration;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.State;

public class CarRentalPolicy extends ActionPolicy
{
	private static final int maximumNumberCars=20;
	private int[] carsToMove;
	
	public CarRentalPolicy()
	{
		carsToMove=new int[maximumNumberCars*maximumNumberCars];
	}
	
	@Override
	public Action getAction(State state) 
	{
		CarRentalState carRentalState=(CarRentalState)state;
		return new CarRentalAction(carsToMove[carRentalState.getID()]);
	}

	@Override
	public void setAction(int state, int action) 
	{
		carsToMove[state]=action;
	}
	
}
