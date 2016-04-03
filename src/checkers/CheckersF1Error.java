package checkers;

import org.apache.commons.math3.linear.ArrayRealVector;

import ReinforcementMachineLearningFramework.LimitedActionsEnvironment;
import ReinforcementMachineLearningFramework.NeuralStatePolicy;
import ReinforcementMachineLearningFramework.State;
import ReinforcementMachineLearningFramework.StateAction;
import evaluationFunctions.EvaluationFunction;
import network.Network;
import network.SplitNetwork;

public class CheckersF1Error extends EvaluationFunction
{

	protected NeuralStatePolicy policy;
	protected LimitedActionsEnvironment environment;
	
	public CheckersF1Error(LimitedActionsEnvironment environment, NeuralStatePolicy policy)
	{
		this.policy=policy;
		this.environment=environment;
	}
	
	@Override
	public double getEval(SplitNetwork network) 
	{
		double truePositives=0.0;
		double falseNegatives=0.0;
		double falsePositives=0.0;
		
		for(State vistedState: policy.getStates())
		{
			StateAction[] possibleStateActions=environment.getAllPossibleStateActions(vistedState);
			
			double bestValue=Double.NEGATIVE_INFINITY;
			StateAction bestStateAction=null;
			
			double bestNetworkValue=Double.NEGATIVE_INFINITY;
			StateAction bestNetworkStateAction=null;
			
			for(StateAction possibleStateAction: possibleStateActions)
			{
				Double value=policy.getStateValueNoNet(possibleStateAction.state);
				if(value!=null)
				{
					double networkValue=network.getOutput(policy.scaleFilterInputs.scaleData(new ArrayRealVector(possibleStateAction.state.getValue()))).getEntry(0);	
					
					value+=policy.rewardFunction.getReward(vistedState, possibleStateAction.state, 
							possibleStateAction.action);
					networkValue+=policy.rewardFunction.getReward(vistedState, possibleStateAction.state, 
							possibleStateAction.action);
					
					if(value>bestValue)
					{
						bestValue=value;
						bestStateAction=possibleStateAction;
					}
					
					if(networkValue>bestNetworkValue)
					{
						bestNetworkValue=value;
						bestNetworkStateAction=possibleStateAction;
					}
				}
			}
			
			if(bestStateAction!=null)
			{
				if(bestStateAction.action.equals(bestNetworkStateAction.action))
				{
					truePositives++;
				}
				else
				{
					falseNegatives++;
					falsePositives++;
				}
			}
		}
		
		double recall=truePositives/(truePositives+falseNegatives);
		double precision=truePositives/(truePositives+falsePositives);
		double f1Score=2.0*recall*precision/(recall+precision);
		
		if(Double.isNaN(f1Score))
		{
			f1Score=Double.POSITIVE_INFINITY;
		}
		
		return f1Score;
	}

}
