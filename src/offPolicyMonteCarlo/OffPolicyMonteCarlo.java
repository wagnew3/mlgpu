package offPolicyMonteCarlo;

import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.ReinforcementLearner;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;

public class OffPolicyMonteCarlo extends ReinforcementLearner
{

	private State[] startStates;
	private ActionPolicy behaviorPolicy;
	private ActionPolicy estimationPolicy;
	private Environment environment;
	private RewardFunction rewardFunction;
	private double discountRate;
	
	private double[][] Qn;
	private double[][] Qd;
	private double[][] Q;
	
	public OffPolicyMonteCarlo(State[] startStates,
			ActionPolicy behaviorPolicy,
			ActionPolicy estimationPolicy,
			Environment environment,
			RewardFunction rewardFunction,
			int numberStates,
			int numberActions,
			double discountRate)
	{
		this.startStates=startStates;
		this.behaviorPolicy=behaviorPolicy;
		this.estimationPolicy=estimationPolicy;
		this.environment=environment;
		this.rewardFunction=rewardFunction;
		this.discountRate=discountRate;
		Qn=new double[numberStates][numberActions];
		Qd=new double[numberStates][numberActions];
		Q=new double[numberStates][numberActions];
	}
	
	@Override
	public void learn() 
	{
		while(true)
		{
			for(int startStateNumber=0; startStateNumber<startStates.length; startStateNumber++)
			{
				makeGreedy(behaviorPolicy);
				makeGreedy(estimationPolicy);
				
				int numberSteps=1;
				environment.setStartState(startStates[startStateNumber]);
				while(environment.step(estimationPolicy)!=-1 && numberSteps<20)
				{
					numberSteps++;
				}
				State estimationTerminationState=environment.getCurrentState();
				
				environment.setStartState(startStates[startStateNumber]);
				while(environment.step(behaviorPolicy)!=-1)
				{
					
				}
				State terminationState=environment.getCurrentState();
				State intermediateState=terminationState;
				
				boolean different=false;
				while(intermediateState.getPreviousState()!=null 
						&& !(different=estimationPolicy.stateActionChance(intermediateState.getPreviousState(), intermediateState.getActionTaken())==0))
				{
					intermediateState=intermediateState.getPreviousState();
				}
				
				/*
				int stateID=intermediateState.getPreviousState().getID();
				int actionID=intermediateState.getActionTaken().getActionID();
				double d=estimationPolicy.stateActionChance(intermediateState.getPreviousState(), intermediateState.getActionTaken());
				*/
				
				if(different)
				{
					double behaviorPolicyChance=1.0;
					double reward=0.0;
					State chanceState=terminationState;
					while(!chanceState.equals(intermediateState))
					{
						if(chanceState.getActionTaken()!=null)
						{
							behaviorPolicyChance*=behaviorPolicy.stateActionChance(chanceState.getPreviousState(), chanceState.getActionTaken());		
						}
						reward+=rewardFunction.getReward(chanceState.getPreviousState(), chanceState, chanceState.getActionTaken())
								+discountRate*reward;
						chanceState=chanceState.getPreviousState();
					}
					if(chanceState.getActionTaken()!=null)
					{
						behaviorPolicyChance*=behaviorPolicy.stateActionChance(chanceState.getPreviousState(), chanceState.getActionTaken());
						chanceState=chanceState.getPreviousState();	
						reward+=rewardFunction.getReward(chanceState.getPreviousState(), chanceState, chanceState.getActionTaken())
								+discountRate*reward;
					}
						int sID=intermediateState.getID();
						int aID=intermediateState.getActionTaken().getActionID();
					try
					{
					Qn[intermediateState.getPreviousState().getID()][intermediateState.getActionTaken().getActionID()]+=behaviorPolicyChance*reward;
					}
					catch(Exception e)
					{
						e.printStackTrace();
					}
					int u=0;
					Qd[intermediateState.getPreviousState().getID()][intermediateState.getActionTaken().getActionID()]+=behaviorPolicyChance;
					Q[intermediateState.getPreviousState().getID()][intermediateState.getActionTaken().getActionID()]
							=Qn[intermediateState.getPreviousState().getID()][intermediateState.getActionTaken().getActionID()]
							/Qd[intermediateState.getPreviousState().getID()][intermediateState.getActionTaken().getActionID()];
				}
			}
		}
	}

	private void makeGreedy(ActionPolicy policy)
	{
		for(int stateNumber=0; stateNumber<Qn.length; stateNumber++)
		{
			int bestAction=-1;
			double bestQ=Double.NEGATIVE_INFINITY;
			for(int actionNumber=0; actionNumber<Qn[0].length; actionNumber++)
			{
				if(Qd[stateNumber][actionNumber]!=0.0 && Qn[stateNumber][actionNumber]/Qd[stateNumber][actionNumber]>=bestQ)
				{
					bestQ=Qn[stateNumber][actionNumber]/Qd[stateNumber][actionNumber];
					bestAction=actionNumber;
				}
			}
			policy.setAction(stateNumber, bestAction);
		}
	}
	
}
