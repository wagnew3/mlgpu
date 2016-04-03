package SP;

import ReinforcementMachineLearningFramework.EGreedyPolicy;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;
import tempDiffLambdaRMLTesuaro.TempDiffLambdaRML;

public class TestSP 
{
	public static void main(String[] args)
	{
		basicSP();
	}
	
	private static void basicSP()
	{
		/*
		int[][] region=new int[][]
		{
			new int[]{0,0,0,0},
			new int[]{0,1,1,1},
			new int[]{0,0,0,0},
			new int[]{0,0,1,0}
		};
		*/
		
		int[][] region=new int[100][100];
		
		SPEnvironment spEnvironment=new SPEnvironment(region, region.length-1, region[0].length-1);
		SPState startState=new SPState(0, 0, region.length-1, region[0].length-1);
		TempDiffLambdaRML tdRML=new TempDiffLambdaRML(spEnvironment, 
				new EGreedyPolicy(0.01),
				startState,
				1000000,
				0.5,
				0.9,
				0.9);
		
		tdRML.learn();

	}
	
}
