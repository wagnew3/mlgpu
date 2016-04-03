package ReinforcementMachineLearningFramework;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.Set;

import org.apache.commons.math3.linear.ArrayRealVector;

import activationFunctions.RectifiedLinearActivationFunction;
import activationFunctions.Sigmoid;
import costFunctions.EuclideanDistanceCostFunction;
import costFunctions.OutputCostFunction;
import network.Network;
import tempDiffLambdaRMLTesuaro.StateAction;
import network.FeedForwardNetwork;
import learning.BackPropGradientDescent;
import learning.BackpropTDRML;

import java.util.Map.Entry;

public class NeuralEGreedyPolicyTesuaro extends EGreedyPolicy
{

	protected int stateLength;
	protected int actionLength;
	public Network network;
	protected Action createAction;
	private double learningRate;
	private double lambda;
	
	Hashtable<State, List<List<Double>>> stateActionValues=new Hashtable<>();
	
	public NeuralEGreedyPolicyTesuaro(double exploreChance, 
			int stateLength, 
			int actionLength, 
			Action createAction, 
			double learningRate,
			double lambda) 
	{
		super(exploreChance);
		this.stateLength=stateLength;
		this.actionLength=actionLength;
		this.createAction=createAction;
		this.learningRate=learningRate;
		this.lambda=lambda;
		network=new FeedForwardNetwork(new RectifiedLinearActivationFunction(), new int[]{stateLength, stateLength, 1});
	}
	
	@Override
	public Action getKthBestAction(State state, int k, Environment environment) 
	{
		if(Math.random()<e)
		{
			Action action=environment.getRandomAction(state);
			ActionListElement actionListElement=new ActionListElement(action, 0.0);
			putActionListElement(state, actionListElement);
			return action;
		}
		
		StateAction[] allPossibleStateActions=((LimitedActionsEnvironment)environment).getAllPossibleStateActions(state);
		StateAction bestStateAction=null;
		double bestValue=Double.NEGATIVE_INFINITY;
		
		/*
		if(stateActionValues.get(state)==null)
		{
			stateActionValues.put(state, new ArrayList<>());
			for(int ind=0; ind<allPossibleActions.length; ind++)
			{
				stateActionValues.get(state).add(new ArrayList<Double>());
			}
		}
		*/
		
		int ind=0;
		for(StateAction possibleStateAction: allPossibleStateActions)
		{
			double value=network.getOutput(new ArrayRealVector(possibleStateAction.getState().getValue())).getEntry(0);
			if(value>bestValue)
			{
				bestStateAction=possibleStateAction;
				bestValue=value;
			}
			
			
			//stateActionValues.get(state).get(ind).add(value);
			ind++;
		}
		
		/*
		System.out.println("State: ");
		for(double stateInt: state.getValue())
		{
			System.out.print(stateInt+", ");
		}
		System.out.println();
		for(List<Double> values: stateActionValues.get(state))
		{
			for(Double value: values)
			{
				System.out.print(value+", ");
			}
			System.out.println();
		}
		System.out.println();
		*/
		
		return bestStateAction.getAction();
	}
	
	protected Action getNNBestAction(State state)
	{
		double[] stateInput=state.getValue();
		double[] rawAction=network.getOutput(new ArrayRealVector(stateInput)).getDataRef();
		Action networkAction=createAction.getFromNNValue(rawAction);
		return networkAction;
	}
	
	public void trainNN(List<StateAction> encounteredStateActions, double delta)
	{
		List<ArrayRealVector> trainStateActions=new ArrayList<>();
		List<ArrayRealVector> values=new ArrayList<>();
		
		for(StateAction stateAction: encounteredStateActions)
		{
			ArrayRealVector stateVector=new ArrayRealVector(stateAction.getState().getValue());
			trainStateActions.add(stateVector);
			values.add(new ArrayRealVector(network.getOutput(stateVector)));
		}
		
		new BackpropTDRML(learningRate, lambda)
			.trainNetwork(network, trainStateActions.toArray(new ArrayRealVector[0]), values.toArray(new ArrayRealVector[0]), new OutputCostFunction(), delta);
	}
	
	protected double[] vectorScale(double a, double[] b)
	{
		double[] result=new double[b.length];
		for(int ind=0; ind<result.length; ind++)
		{
			result[ind]=a*b[ind];
		}
		return result;
	}
	
	protected double[] vectorSum(double[] a, double[] b)
	{
		double[] result=new double[a.length];
		for(int ind=0; ind<result.length; ind++)
		{
			result[ind]=a[ind]+b[ind];
		}
		return result;
	}
	
	protected double[] vectorDifference(double[] a, double[] b)
	{
		double[] result=new double[a.length];
		for(int ind=0; ind<result.length; ind++)
		{
			result[ind]=a[ind]-b[ind];
		}
		return result;
	}
	
	protected double dotProduct(double[] a, double[] b)
	{
		double result=0.0;
		for(int ind=0; ind<a.length; ind++)
		{
			result+=a[ind]*b[ind];
		}
		return result;
	}
	

}
