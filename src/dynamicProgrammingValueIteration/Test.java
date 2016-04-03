package dynamicProgrammingValueIteration;

import dynamicProgrammingPolicyIteration.DPPolicyIterationReinformentLearner;
import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;

public class Test 
{
	
	public static void main(String[] args)
	{
		//testDPVI();
		
		testDPVI();
	}
	
	private static void testDPRL()
	{
		DPPolicyIterationReinformentLearner dprl=new DPPolicyIterationReinformentLearner(101, 
				101,
				new CoinFlipEnviroment(0.4), 
				new CoinFlipReward(),
				new CoinFlipPolicy(101),
				new CoinFlipState(0),
				new CoinFlipState(0),
				new CoinFlipAction(0),
				0.0000000000000000000000001,
				1.0);
		dprl.learn();
		int i=0;
	}
	
	private static void testDPVI()
	{
		DynamicProgrammingValueIteration dpvi=new DynamicProgrammingValueIteration(101, 
				101,
				new CoinFlipEnviroment(0.4), 
				new CoinFlipReward(),
				new CoinFlipPolicy(101),
				new CoinFlipState(0),
				new CoinFlipState(0),
				new CoinFlipAction(0),
				0.0000000000000000000000001,
				1.0);
		dpvi.learn();
		int i=0;
	}

}
