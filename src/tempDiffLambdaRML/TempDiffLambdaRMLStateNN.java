package tempDiffLambdaRML;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.commons.math3.linear.ArrayRealVector;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.ActionListElement;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.LimitedActionsEnvironment;
import ReinforcementMachineLearningFramework.NeuralEGreedyPolicyTesuaro;
import ReinforcementMachineLearningFramework.NeuralEGreedyStatePolicy;
import ReinforcementMachineLearningFramework.NeuralEGreedyStatePolicyTrainOnKVisited;
import ReinforcementMachineLearningFramework.NeuralStatePolicy;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.ReinforcementLearner;
import ReinforcementMachineLearningFramework.RewardFunction;
import ReinforcementMachineLearningFramework.State;
import ReinforcementMachineLearningFramework.StateAction;
import ReinforcementMachineLearningFramework.StatePolicy;
import activationFunctions.Sigmoid;
import costFunctions.EuclideanDistanceCostFunction;
import hc.HCState;
import layer.FullyConnectedLayer;
import layer.InputLayer;
import layer.Layer;
import network.FeedForwardNetwork;
import network.Network;

public class TempDiffLambdaRMLStateNN extends ReinforcementLearner
{

	LimitedActionsEnvironment enviroment;
	NeuralStatePolicy policy;
	State startState;
	int numberEpisodes;
	double stepSize; //a
	double discountRate; //y
	double traceDecay; //lambda
	Network network;
	
	
	public TempDiffLambdaRMLStateNN(LimitedActionsEnvironment enviroment, 
			NeuralStatePolicy policy,
			State startState,
			int numberEpisodes,
			double stepSize,
			double discountRate,
			double traceDecay,
			int stateLength)
	{
		this.enviroment=enviroment;
		this.policy=policy;
		this.startState=startState;
		this.numberEpisodes=numberEpisodes;
		this.stepSize=stepSize;
		this.discountRate=discountRate;
		this.traceDecay=traceDecay;
		
		InputLayer inputLayer=new InputLayer(null, stateLength);
		FullyConnectedLayer hiddenLayer1=new FullyConnectedLayer(new Sigmoid(), inputLayer, stateLength);
		FullyConnectedLayer hiddenLayer2=new FullyConnectedLayer(new Sigmoid(), hiddenLayer1, stateLength/2);
		FullyConnectedLayer outputLayer=new FullyConnectedLayer(new Sigmoid(), hiddenLayer2, 1);
		network=new FeedForwardNetwork(new Layer[]{inputLayer, hiddenLayer1, hiddenLayer2, outputLayer});
	}
	
	public void learn() 
	{
		Hashtable<State, Double> eligibilityTraces;
		for(int numberEpisodesCompleted=0; numberEpisodesCompleted<numberEpisodes; numberEpisodesCompleted++)
		{
			enviroment.setStartState(startState);
			State currentState=enviroment.getCurrentState();
			Action bestAction=policy.getAction(currentState, enviroment);
			
			int actions=0;
			
			eligibilityTraces=new Hashtable<>(); //[0]=StateAction, [1]=ActionListElement
			double reward=0.0;
			
			while(!enviroment.getCurrentState().isEndState())
			{
				actions++;
				reward=enviroment.takeAction(bestAction, 0);
				State newCurrentState=enviroment.getCurrentState();
				Action newBestAction=policy.getAction(newCurrentState, enviroment);
				
				
				setStateEligibility(currentState, getStateEligibility(currentState, eligibilityTraces)+1, eligibilityTraces);
				
				List<State> toRemove=new ArrayList<>();
				synchronized(policy)
				{
					for(State state: eligibilityTraces.keySet())
					{
						double diff=reward+discountRate*getStateValue(newCurrentState)-getStateValue(state);
						if(diff>0 && getStateValue(state)>0.5)
						{
							int u=0;
						}
						policy.setStateValue(state, policy.getStateValue(state)
								+stepSize*diff*getStateEligibility(state, eligibilityTraces));
						setStateEligibility(state, discountRate*traceDecay*getStateEligibility(state, eligibilityTraces), eligibilityTraces);
						if(getStateEligibility(state, eligibilityTraces)<0.001)
						{
							toRemove.add(state);
						}
					}
				}
				
				for(State state: toRemove)
				{
					eligibilityTraces.remove(state);
				}
				
				
				
				currentState=newCurrentState;
				bestAction=newBestAction;
			}
			
			if(numberEpisodesCompleted%10000==0)
			{
				((NeuralEGreedyStatePolicyTrainOnKVisited)policy).trainNN();
			}
			
			System.out.println("Took "+actions+" actions Episode "+numberEpisodesCompleted);
		}
	}
	
	private double getStateValue(State state)
	{
		return policy.getStateValue(state);
	}
	
	private void setStateValue(State state, double value)
	{
		policy.setStateValue(state, value);
	}
	
	private double getStateEligibility(State state, Hashtable<State, Double> eligibilityTraces)
	{
		Double value=eligibilityTraces.get(state);
		if(value==null)
		{
			return 0.0;
		}
		else
		{
			return value;
		}
	}
	
	private void setStateEligibility(State state, double eligibility, Hashtable<State, Double> eligibilityTraces)
	{
		eligibilityTraces.put(state, eligibility);
	}

}
