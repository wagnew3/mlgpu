package dynamicProgrammingPolicyIteration;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;

public class CarRentalRewardFunction extends RewardFunction
{
	
	private static final int rentProfit=10;
	private static final int moveCost=2;
	
	public CarRentalRewardFunction()
	{
		
	}

	@Override
	public double getReward(State previousState, State currentState, Action action) 
	{
		return rentProfit*(((CarRentalState)previousState).numberCarsLocation1
				+((CarRentalState)previousState).numberCarsLocation2
				+((CarRentalState)previousState).numberCarsReturnedLocation1
				+((CarRentalState)previousState).numberCarsReturnedLocation2
				-((CarRentalState)currentState).numberCarsLocation1
				-((CarRentalState)currentState).numberCarsLocation2)
				-moveCost*Math.abs(((CarRentalAction)action).getNumberCarsToMove());
	}

}
