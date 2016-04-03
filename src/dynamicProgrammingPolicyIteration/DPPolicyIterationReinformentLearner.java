package dynamicProgrammingPolicyIteration;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.ReinforcementLearner;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;

public class DPPolicyIterationReinformentLearner extends ReinforcementLearner
{
	
	private double[] stateValues;
	private int numberStates;
	private int numberActions;
	private Environment enviroment;
	private RewardFunction rewardFunction;
	private ActionPolicy policy;
	private State stateA;
	private State stateB;
	private Action action;
	private double tolerance;
	private double rewardDiscount;
	
	public DPPolicyIterationReinformentLearner(int numberStates, 
			int numberActions,
			Environment enviroment, 
			RewardFunction rewardFunction,
			ActionPolicy policy,
			State stateA,
			State stateB,
			Action action,
			double tolerance,
			double rewardDiscount)
	{
		stateValues=new double[numberStates];
		this.numberStates=numberStates;
		this.numberActions=numberActions;
		this.enviroment=enviroment;
		this.rewardFunction=rewardFunction;
		this.policy=policy;
		this.stateA=stateA;
		this.stateB=stateB;
		this.action=action;
		this.tolerance=tolerance;
		this.rewardDiscount=rewardDiscount;
	}

	@Override
	public void learn() 
	{
		boolean policyStable=false;
		
		while(!policyStable)
		{
			policyStable=true;
			
			double maxDifference;
			do
			{
				maxDifference=0.0;
				for(int oldStateNumber=0; oldStateNumber<numberStates; oldStateNumber++)
				{
					stateA.createFromID(oldStateNumber);
					if(oldStateNumber==100)
					{
						int i=0;
					}
					double stateValueSum=0.0;
					for(int newStateNumber=0; newStateNumber<numberStates; newStateNumber++)
					{
						stateB.createFromID(newStateNumber);
						double transitionProbability=
								enviroment.transistionProbability(stateA, stateB, policy.getAction(stateA));
						double reward=rewardFunction.getReward(stateA, stateB, policy.getAction(stateA));
						double stateBStateValue=stateValues[stateB.getID()];
						if(transitionProbability>0.0)
						{
							int u=0;
						}
						stateValueSum+=transitionProbability*(reward+rewardDiscount*stateBStateValue);
					}
					if(stateValueSum>0)
					{
						int u=0;
					}
					maxDifference=Math.max(maxDifference, Math.abs(stateValues[stateA.getID()]-stateValueSum));
					stateValues[stateA.getID()]=stateValueSum;
				}
			}
			while(maxDifference>tolerance);
			
			int u=0;
			
			for(int oldStateNumber=0; oldStateNumber<numberStates; oldStateNumber++)
			{
				stateA.createFromID(oldStateNumber);
				int bestActionNumber=-1;
				double bestActionReward=Double.NEGATIVE_INFINITY;
				for(int actionNumber=0; actionNumber<numberActions; actionNumber++)
				{
					double currentActionReward=0.0;
					action.setActionFromID(actionNumber);
					for(int newStateNumber=0; newStateNumber<numberStates; newStateNumber++)
					{
						stateB.createFromID(newStateNumber);
						double transitionProbability=
								enviroment.transistionProbability(stateA, stateB, action);
						double reward=rewardFunction.getReward(stateA, stateB, policy.getAction(stateA));
						double stateBStateValue=stateValues[stateB.getID()];
						currentActionReward+=transitionProbability*(reward+rewardDiscount*stateBStateValue);
					}
					if(currentActionReward>bestActionReward)
					{
						bestActionReward=currentActionReward;
						bestActionNumber=actionNumber;
					}
				}
				if(policy.getAction(stateA).getActionID()!=bestActionNumber)
				{
					policyStable=false;
				}
				policy.setAction(stateA.getID(), bestActionNumber);
			}
		}
	}

}
