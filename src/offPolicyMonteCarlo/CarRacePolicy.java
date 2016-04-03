package offPolicyMonteCarlo;

import org.apache.commons.math3.distribution.NormalDistribution;

import dynamicProgrammingPolicyIteration.CarRentalAction;
import dynamicProgrammingPolicyIteration.CarRentalState;
import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.State;

public class CarRacePolicy extends ActionPolicy
{

	public int[][] policies;
	
	private NormalDistribution normalDistribution=null;
	
	public CarRacePolicy(int numberX, int numberY, int numberXVel, int numberYVel, double randomnessMean, double randomnessSD)
	{
		policies=new int[numberX*numberY*numberXVel*numberYVel][2];
		if(randomnessSD>0)
		{
			normalDistribution=new NormalDistribution(randomnessMean, randomnessSD);
		}
	}
	
	@Override
	public Action getAction(State state) 
	{
		CarRaceState carRaceState=(CarRaceState)state;
		if(normalDistribution!=null)
		{
			int xVelPerturbation=(int)Math.round(normalDistribution.sample());
			int yVelPerturbation=(int)Math.round(normalDistribution.sample());
			return new CarRaceAction(policies[carRaceState.getID()][0]+xVelPerturbation,
							     policies[carRaceState.getID()][1]+yVelPerturbation);
		}
		else
		{
			return new CarRaceAction(policies[carRaceState.getID()][0], policies[carRaceState.getID()][1]);
			
		}
	}

	@Override
	public void setAction(int state, int action) 
	{
		CarRaceAction carRaceAction=new CarRaceAction(action);
		policies[state]=new int[]{carRaceAction.xAccel, carRaceAction.yAccel};
	}

	@Override
	public double stateActionChance(State state, Action action) 
	{
		if(action==null)
		{
			return 1.0;
		}
		CarRaceState carRaceState=(CarRaceState)state;
		CarRaceAction carRaceAction=(CarRaceAction)action;
		
		int actionID=carRaceAction.getActionID();
		if(normalDistribution==null)
		{
			if(policies[carRaceState.getID()][0]==carRaceAction.xAccel
				&& policies[carRaceState.getID()][1]==carRaceAction.yAccel)
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
			return normalDistribution.probability(policies[carRaceState.getID()][0]-carRaceAction.xAccel-0.5,
					policies[carRaceState.getID()][0]-carRaceAction.xAccel+0.5)
					*normalDistribution.probability(policies[carRaceState.getID()][1]-carRaceAction.yAccel-0.5,
							policies[carRaceState.getID()][1]-carRaceAction.yAccel+0.5);
			
		}
	}

}
