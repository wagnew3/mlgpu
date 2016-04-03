package hc;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.EGreedyPolicy;
import ReinforcementMachineLearningFramework.EGreedyStatePolicy;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.NeuralEGreedyPolicy;
import ReinforcementMachineLearningFramework.NeuralEGreedyPolicyTesuaro;
import ReinforcementMachineLearningFramework.NeuralEGreedyStatePolicy;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;
import tempDiffLambdaRML.TempDiffLambdaRMLNN;
import tempDiffLambdaRML.TempDiffLambdaRMLState;
import tempDiffLambdaRML.TempDiffLambdaRMLStateNN;
import tempDiffLambdaRMLTesuaro.TempDiffLambdaRML;
import tempDiffLambdaRMLTesuaro.TempDiffLambdaRMLNNTesuaro;

public class TestHC 
{
	public static void main(String[] args)
	{
		//noMLHC();
		//stateNOMLHC();
		//MLHCTesuaro();
		//MLHC();
		stateMLHC();
	}
	
	private static void noMLHC()
	{
		int numberBits=10;
		int numberZero=3;
		int[] start=new int[numberBits];
		for(int ind=0; ind<start.length; ind++)
		{
			start[ind]=1;
		}
		
		HCEnvironment spEnvironment=new HCEnvironment(numberBits, start, numberZero);
		HCState startState=new HCState(start, numberZero);
		TempDiffLambdaRML tdRML=new TempDiffLambdaRML(spEnvironment, 
				new EGreedyPolicy(0.01),
				startState,
				10000000,
				0.25,
				0.9,
				0.9);
		
		tdRML.learn();

	}
	
	private static void stateNOMLHC()
	{
		int numberBits=20;
		int numberZero=5;
		int[] start=new int[numberBits];
		for(int ind=0; ind<start.length; ind++)
		{
			start[ind]=1;
		}
		
		HCEnvironment spEnvironment=new HCEnvironment(numberBits, start, numberZero);
		HCState startState=new HCState(start, numberZero);
		TempDiffLambdaRMLState tdRMLS=new TempDiffLambdaRMLState(spEnvironment, 
				new EGreedyStatePolicy(0.01, new HCRewardFunction(numberZero)),
				startState,
				1000000,
				0.1,
				0.9,
				0.9);
		
		tdRMLS.learn();
		System.out.println("Beginning Runs From Random States");
		int totalActions=0;
		for(int run=0; run<1000; run++)
		{
			totalActions+=tdRMLS.run(getRandomHCState(numberBits, numberZero));
		}
		System.out.println(totalActions);

	}
	
	private static void MLHCTesuaro()
	{
		int numberBits=4;
		int numberZero=1;
		int[] start=new int[numberBits];
		for(int ind=0; ind<start.length; ind++)
		{
			start[ind]=1;
		}
		
		double lambda=0.1;
		
		HCEnvironment spEnvironment=new HCEnvironment(numberBits, start, numberZero);
		HCState startState=new HCState(start, numberZero);
		TempDiffLambdaRMLNNTesuaro tdRML=new TempDiffLambdaRMLNNTesuaro(spEnvironment, 
				new NeuralEGreedyPolicyTesuaro(0.1, 
						numberBits, 
						numberBits, 
						new HCAction(new int[0]), 
						3.0,
						lambda),
				startState,
				1000000,
				0.25,
				0.9,
				lambda);
		tdRML.learn();
	}
	
	private static void MLHC()
	{
		int numberBits=10;
		int numberZero=5;
		int[] start=new int[numberBits];
		for(int ind=0; ind<start.length; ind++)
		{
			start[ind]=1;
		}
		
		double lambda=0.1;
		
		HCEnvironment spEnvironment=new HCEnvironment(numberBits, start, numberZero);
		HCState startState=new HCState(start, numberZero);
		TempDiffLambdaRMLNN tdRML=new TempDiffLambdaRMLNN(spEnvironment, 
				new NeuralEGreedyPolicy(0.1, 
						numberBits, 
						numberBits, 
						new HCAction(new int[0])),
				startState,
				1000000,
				0.1,
				0.9,
				lambda);
		tdRML.learn();
	}
	
	private static void stateMLHC()
	{
		int numberBits=20;
		int numberZero=5;
		int[] start=new int[numberBits];
		for(int ind=0; ind<start.length; ind++)
		{
			start[ind]=1;
		}
		
		HCEnvironment spEnvironment=new HCEnvironment(numberBits, start, numberZero);
		HCState startState=new HCState(start, numberZero);
		TempDiffLambdaRMLStateNN tdRMLS=new TempDiffLambdaRMLStateNN(spEnvironment, 
				new NeuralEGreedyStatePolicy(0.05, new HCRewardFunction(numberZero), numberBits),
				startState,
				200,
				0.1,
				0.9,
				0.9);
		tdRMLS.learn();
		
		System.out.println("Beginning Runs From Random States");
		int totalActions=0;
		for(int run=0; run<1000; run++)
		{
			totalActions+=tdRMLS.run(getRandomHCState(numberBits, numberZero));
		}
		System.out.println(totalActions);

	}
	
	private static HCState getRandomHCState(int numberBits, int numberZero)
	{
		int[] stateData=new int[numberBits];
		for(int ind=0; ind<stateData.length; ind++)
		{
			stateData[ind]=(int)Math.round(Math.random());
		}
		return new HCState(stateData, numberZero);
	}
	
}
