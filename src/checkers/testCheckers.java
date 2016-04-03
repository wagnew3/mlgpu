package checkers;

import java.io.File;

import ReinforcementMachineLearningFramework.EGreedyPolicy;
import ReinforcementMachineLearningFramework.EGreedyStatePolicy;
import ReinforcementMachineLearningFramework.NeuralEGreedyPolicy;
import ReinforcementMachineLearningFramework.NeuralEGreedyPolicyTesuaro;
import ReinforcementMachineLearningFramework.NeuralEGreedyStatePolicy;
import ReinforcementMachineLearningFramework.NeuralEGreedyStatePolicyTrainOnKVisited;
import ReinforcementMachineLearningFramework.NeuralMCTSPolicy;
import ReinforcementMachineLearningFramework.NeuralStatePolicy;
import ReinforcementMachineLearningFramework.StatePolicy;
import hc.HCAction;
import hc.HCEnvironment;
import hc.HCRewardFunction;
import hc.HCState;
import monteCarloState.MonteCarloStateNNGame;
import tempDiffLambdaRML.TempDiffLambdaEMLStateNNGame;
import tempDiffLambdaRML.TempDiffLambdaRMLNN;
import tempDiffLambdaRML.TempDiffLambdaRMLState;
import tempDiffLambdaRML.TempDiffLambdaRMLStateNN;
import tempDiffLambdaRMLTesuaro.TempDiffLambdaRML;
import tempDiffLambdaRMLTesuaro.TempDiffLambdaRMLNNTesuaro;

public class testCheckers 
{
	
	protected static final File policySaveLocationA=new File("/home/c/workspace/Reinforcement Machine Learning/SavedNeuralNets/policyA");
	protected static final File policySaveLocationB=new File("/home/c/workspace/Reinforcement Machine Learning/SavedNeuralNets/policyB");
	
	public static void main(String[] args)
	{
		//noMLHC();
		//stateNOMLHC();
		//MLHCTesuaro();
		//MLHC();
		//stateMLHC();
		//stateMLHCTestHumanPlay();
		//stateMLMCTSTestHumanPlay();
		stateMCRMLMLMCTSTestHumanPlay();
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
	
	private static void stateMLHC()
	{
		int size=4;
		
		byte[][] startBoard=getStartState(size);
		
		CheckersEnvironment checkersEnvironment=new CheckersEnvironment(size, new CheckersReward());
		CheckersState startState=new CheckersState(startBoard);
		TempDiffLambdaEMLStateNNGame tdRMLS=new TempDiffLambdaEMLStateNNGame(checkersEnvironment, 
				new NeuralEGreedyStatePolicy[]
				{
					new NeuralEGreedyStatePolicy(0.1, new CheckersReward(), size*size),
					new NeuralEGreedyStatePolicy(0.1, new CheckersReward(), size*size)
				},
				startState,
				1000,
				0.25,
				0.9,
				0.9,
				new CheckersReward());
		tdRMLS.learn(2);
		/*
		System.out.println("Beginning Runs From Random States");
		int totalActions=0;
		for(int run=0; run<1000; run++)
		{
			totalActions+=tdRMLS.run(getRandomHCState(numberBits, numberZero));
		}
		System.out.println(totalActions);
		*/

	}
	
	private static void stateMLHCTestHumanPlay()
	{
		int size=6;
		
		byte[][] startBoard=getStartState(size);
		
		CheckersEnvironment checkersEnvironment=new CheckersEnvironment(size, new CheckersReward());
		CheckersState startState=new CheckersState(startBoard);
		TempDiffLambdaEMLStateNNGame tdRMLS=new TempDiffLambdaEMLStateNNGame(checkersEnvironment, 
				new NeuralEGreedyStatePolicy[]
				{
					new NeuralEGreedyStatePolicyTrainOnKVisited(0.1, new CheckersReward(), size*size, 20),
					new NeuralEGreedyStatePolicyTrainOnKVisited(0.1, new CheckersReward(), size*size, 20)
				},
				startState,
				50001,
				0.25,
				0.9,
				0.9,
				new CheckersReward());
		tdRMLS.learn(2);
		while(true)
		{
			try
			{
				tdRMLS.playAgainstHuman(2, 0);
			}
			catch(Exception e)
			{
				e.printStackTrace();
			}
			int u=0;
		}
	}

	private static void stateMLMCTSTestHumanPlay()
	{
		int size=4;
		
		byte[][] startBoard=getStartState(size);
		
		CheckersEnvironment checkersEnvironment=new CheckersEnvironment(size, new CheckersReward());
		CheckersState startState=new CheckersState(startBoard);
		TempDiffLambdaEMLStateNNGame tdRMLS=new TempDiffLambdaEMLStateNNGame(checkersEnvironment, 
				new NeuralStatePolicy[]
				{
					new NeuralMCTSPolicy(size*size, new CheckersReward(), checkersEnvironment),
					new NeuralMCTSPolicy(size*size, new CheckersReward(), checkersEnvironment)
				},
				startState,
				100001,
				0.25,
				0.9,
				0.9,
				new CheckersReward());
		tdRMLS.learn(2);
		while(true)
		{
			try
			{
				tdRMLS.playAgainstHuman(2, 0);
			}
			catch(Exception e)
			{
				e.printStackTrace();
			}
			int u=0;
		}
	}
	
	private static void stateMCRMLMLMCTSTestHumanPlay()
	{
		int size=8;
		byte[][] startBoard=getStartState(size);
		
		CheckersEnvironment checkersEnvironment=new CheckersEnvironment(size, new CheckersReward());
		CheckersState startState=new CheckersState(startBoard);
		MonteCarloStateNNGame tdRMLS=new MonteCarloStateNNGame(checkersEnvironment, 
				new NeuralStatePolicy[]
				{

					//(NeuralStatePolicy)StatePolicy.loadPolicy(policySaveLocationA),
					//(NeuralStatePolicy)StatePolicy.loadPolicy(policySaveLocationB)
					
					
					new NeuralMCTSPolicy(size*size, new CheckersReward(), checkersEnvironment),
					new NeuralMCTSPolicy(size*size, new CheckersReward(), checkersEnvironment)
					
					
				},
				startState,
				2000,
				new CheckersReward());
		tdRMLS.learn(2);
		while(true)
		{
			try
			{
				tdRMLS.playAgainstHuman(2, 0);
			}
			catch(Exception e)
			{
				e.printStackTrace();
			}
			int u=0;
		}
	}
	
	private static byte[][] getStartState(int size)
	{
		if(size<4)
		{
			System.out.println("Size must be at least 4!");
		}
		if(size%2==1)
		{
			System.out.println("Size must be even!");
		}
		
		byte[][] startBoard=new byte[size][size];
		
		for(int rowInd=0; rowInd<startBoard.length/2-1; rowInd++)
		{
			for(int colInd=0; colInd<startBoard[0].length; colInd++)
			{
				if((colInd+rowInd)%2==1)
				{
					startBoard[rowInd][colInd]=-1;
				}
			}
		}
		
		for(int rowInd=startBoard.length-1; rowInd>startBoard.length/2; rowInd--)
		{
			for(int colInd=0; colInd<startBoard[0].length; colInd++)
			{
				if((colInd+rowInd)%2==1)
				{
					startBoard[rowInd][colInd]=1;
				}
			}
		}
		return startBoard;
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
