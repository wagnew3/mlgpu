package cardGame4540;

import offPolicyMonteCarlo.CarRaceAction;
import offPolicyMonteCarlo.CarRaceState;

import org.apache.commons.math3.distribution.NormalDistribution;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.State;

public class CardGamePolicy extends ActionPolicy
{
	
	int[] policies;
	
	private NormalDistribution normalDistribution=null;

	public CardGamePolicy(int maxNumberRed, int maxNumberBlack, double randomnessMean, double randomnessSD)
	{
		policies=new int[(maxNumberRed+1)*(maxNumberBlack+1)];
		for(int index=0; index<policies.length; index++)
		{
			if(index%27>4+index/27)
			{
				policies[index]=1;
			}
		}
		if(randomnessSD>0.0)
		{
			normalDistribution=new NormalDistribution(randomnessMean, randomnessSD);
		}
	}
	
	@Override
	public Action getAction(State state) 
	{
		CardGameState cardGameState=(CardGameState)state;
		if(normalDistribution!=null)
		{
			int perturbation=(int)Math.round(normalDistribution.sample());
			if(perturbation>=1)
			{
				return new CardGameAction(1);
			}
			else if(perturbation<=-1)
			{
				return new CardGameAction(0);
			}
			else
			{
				return new CardGameAction(policies[cardGameState.getID()]);
			}
		}
		else
		{
			return new CardGameAction(policies[cardGameState.getID()]);
		}
	}

	@Override
	public void setAction(int state, int action) 
	{
		policies[state]=action;
	}

	@Override
	public double stateActionChance(State state, Action action) 
	{
		if(action==null)
		{
			return 1.0;
		}
		CardGameState cardGameState=(CardGameState)state;
		CardGameAction cardGameAction=(CardGameAction)action;
		
		if(normalDistribution==null)
		{
			if(policies[cardGameState.getID()]==cardGameAction.action)
			{
				return 1.0;
			}
			else
			{
				return 0.0;
			}
		}
		else
		{
			if(cardGameAction.action-policies[cardGameState.getID()]>=1)
			{
				return normalDistribution.probability(0.5, Double.POSITIVE_INFINITY);
			}
			else if(cardGameAction.action-policies[cardGameState.getID()]<=1)
			{
				return normalDistribution.probability(Double.NEGATIVE_INFINITY, -0.5);
			}
			else
			{
				return normalDistribution.probability(-0.5, 0.5);
			}	
		}
	}

}
