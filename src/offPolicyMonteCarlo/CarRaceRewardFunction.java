package offPolicyMonteCarlo;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;

public class CarRaceRewardFunction extends RewardFunction
{

	private CarRaceEnvironment carRaceEnvironment;
	
	public CarRaceRewardFunction(CarRaceEnvironment carRaceEnvironment)
	{
		this.carRaceEnvironment=carRaceEnvironment;
	}
	
	@Override
	public double getReward(State previousState, State currentState, Action action) 
	{
		CarRaceState currentCarRaceState=(CarRaceState)currentState;
		if(!carRaceEnvironment.isOnTrack(currentCarRaceState))
		{
			return -5.0;
		}
		else if(carRaceEnvironment.track[currentCarRaceState.x][currentCarRaceState.y]==1)
		{
			return 1.0;
		}
		else
		{
			return -1.0;
		}
	}

}
