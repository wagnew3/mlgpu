package cardGame4540;

import dynamicProgrammingValueIteration.CoinFlipAction;
import dynamicProgrammingValueIteration.CoinFlipEnviroment;
import dynamicProgrammingValueIteration.CoinFlipPolicy;
import dynamicProgrammingValueIteration.CoinFlipReward;
import dynamicProgrammingValueIteration.CoinFlipState;
import dynamicProgrammingValueIteration.DynamicProgrammingValueIteration;
import offPolicyMonteCarlo.CarRaceEnvironment;
import offPolicyMonteCarlo.CarRacePolicy;
import offPolicyMonteCarlo.CarRaceRewardFunction;
import offPolicyMonteCarlo.CarRaceState;
import offPolicyMonteCarlo.OffPolicyMonteCarlo;
import ReinforcementMachineLearningFramework.State;

public class Run 
{
	
	public static void main(String[] args)
	{
		//findBestStrategy();
		findBestStrategyDPVI();
	}
	
	private static void findBestStrategy()
	{
		CardGameEnvironment cardGameEnvironment=new CardGameEnvironment();
		OffPolicyMonteCarlo opmc=new OffPolicyMonteCarlo(new State[]{new CardGameState(26, 26, null, null)},
				new CardGamePolicy(26, 26, 0.0, 0.25),
				new CardGamePolicy(26, 26, 0.0, 0.0),
				cardGameEnvironment,
				new CardGameRewardFunction(),
				(26+1)*(26+1),
				2,
				1.0);
		opmc.learn();
		int u=0;
	}
	
	private static void findBestStrategyDPVI()
	{
		DynamicProgrammingValueIteration dpvi=new DynamicProgrammingValueIteration((26+1)*(26+1),
				2,
				new CardGameEnvironment(), 
				new CardGameRewardFunction(),
				new CardGamePolicy(26, 26, 0.0, 0.01),
				new CardGameState(26, 26, null, null),
				new CardGameState(26, 26, null, null),
				new CardGameAction(0),
				0.01,
				1.0);
		dpvi.learn();
		int i=0;
	}

}
