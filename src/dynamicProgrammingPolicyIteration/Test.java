package dynamicProgrammingPolicyIteration;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;

public class Test 
{
	
	public static void main(String[] args)
	{
		testCarRental();
	}
	
	private static void testCarRental()
	{
		DPPolicyIterationReinformentLearner learner=new DPPolicyIterationReinformentLearner(400, 
				10,
				new CarRentalEnvironment(), 
				new CarRentalRewardFunction(),
				new CarRentalPolicy(),
				new CarRentalState(0, 0, null),
				new CarRentalState(0, 0, null),
				new CarRentalAction(0),
				0.1,
				0.9);
		learner.learn();
		int u=0;
	}

}
