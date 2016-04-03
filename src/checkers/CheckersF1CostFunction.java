package checkers;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.commons.math3.linear.ArrayRealVector;

import ReinforcementMachineLearningFramework.LimitedActionsEnvironment;
import ReinforcementMachineLearningFramework.NeuralStatePolicy;
import ReinforcementMachineLearningFramework.Policy;
import ReinforcementMachineLearningFramework.State;
import ReinforcementMachineLearningFramework.StateAction;
import ReinforcementMachineLearningFramework.NeuralMCTSPolicy;
import costFunctions.CostFunction;
import costFunctions.PreprocessCostFunction;
import network.Network;
import network.SplitNetwork;

public class CheckersF1CostFunction extends PreprocessCostFunction
{

	protected NeuralStatePolicy policy;
	protected State typeState;
	protected LimitedActionsEnvironment environment;
	protected HashMap<DoubleWrapper, Double> derivatives;
	protected double totalCost;
	protected double exponent=3.0;
	
	public CheckersF1CostFunction(LimitedActionsEnvironment environment, NeuralStatePolicy policy, State typeState)
	{
		this.environment=environment;
		this.policy=policy;
		this.typeState=typeState;
	}
	
	@Override
	public double getCost(ArrayRealVector input, ArrayRealVector networkOutput,
			ArrayRealVector desiredOutput) 
	{
		State inputState=typeState.getFromNNValue(input.getDataRef());
		return 0.0;
	}
	
	@Override
	public double totalCost()
	{
		return totalCost;
	}

	@Override
	public ArrayRealVector getCostDerivative(ArrayRealVector input,
			ArrayRealVector networkOutput, ArrayRealVector desiredOutput) 
	{
		return new ArrayRealVector(new double[]{getDerivative(input.getDataRef())});
	}

	@Override
	public void preprocessDerivatives(ArrayRealVector[] inputs,
			ArrayRealVector[] desiredOutputs, SplitNetwork network) 
	{
		derivatives=new HashMap<>();
		
		for(Entry<State, double[]> stateEntry: ((NeuralMCTSPolicy)policy).stateInfo.entrySet())
		{
			if(stateEntry.getValue()[1]>25)
			{
				StateAction[] possibleStateActions=environment.getAllPossibleStateActions(stateEntry.getKey());
				
				double[] stateValues=new double[possibleStateActions.length];
				double totalStateValue=0.0;
				double[] networkStateValues=new double[possibleStateActions.length];
				double totalNetworkStateValue=0.0;
				
				for(int stateActionInd=0; stateActionInd<possibleStateActions.length; stateActionInd++)
				{
					StateAction possibleStateAction=possibleStateActions[stateActionInd];
					Double value=policy.getStateValueNoNet(possibleStateAction.state);
					if(value!=null)
					{
						value=policy.scaleFilterOutputs.scaleData(new ArrayRealVector(new double[]{value})).getEntry(0);
						double networkValue=network.getOutput(policy.scaleFilterInputs.scaleData(new ArrayRealVector(possibleStateAction.state.getValue()))).getEntry(0);	
						
						value+=policy.rewardFunction.getReward(stateEntry.getKey(), possibleStateAction.state, 
								possibleStateAction.action);
						networkValue+=policy.rewardFunction.getReward(stateEntry.getKey(), possibleStateAction.state, 
								possibleStateAction.action);
						
						stateValues[stateActionInd]=Math.pow(value, exponent);
						totalStateValue+=stateValues[stateActionInd];
						
						networkStateValues[stateActionInd]=Math.pow(networkValue, exponent);
						totalNetworkStateValue+=networkStateValues[stateActionInd];
					}
				}
				
				if(totalStateValue!=0 && totalNetworkStateValue!=0)
				{
					double[] derivatives=new double[possibleStateActions.length];
					for(int derivativeInd=0; derivativeInd<derivatives.length; derivativeInd++)
					{
						derivatives[derivativeInd]=0;
						for(int stateInd=0; stateInd<stateValues.length; stateInd++)
						{
							double term=stateValues[stateInd]/totalStateValue-networkStateValues[stateInd]/totalNetworkStateValue;
							if(stateInd==derivativeInd)
							{
								derivatives[derivativeInd]+=(stateValues[stateInd]/totalStateValue)
										-3.0*Math.pow(networkStateValues[derivativeInd], 0.666666667)/totalNetworkStateValue;
								derivatives[derivativeInd]+=getDevTotal(stateValues, networkStateValues, derivativeInd);
							}
							else
							{
								derivatives[derivativeInd]+=(stateValues[stateInd]/totalStateValue)
										+networkStateValues[derivativeInd]
												*getDevTotal(stateValues, networkStateValues, derivativeInd);
							}
							if(term<0)
							{
								derivatives[derivativeInd]=-derivatives[derivativeInd];
							}
						}
					}
					
					for(int derivativeInd=0; derivativeInd<derivatives.length; derivativeInd++)
					{
						incDerivative(policy.scaleFilterInputs.scaleData(
								new ArrayRealVector(possibleStateActions[derivativeInd].state.getValue())).getDataRef(),
								derivatives[derivativeInd]);
					}
				}
			}
		}
	}
	
	@Override
	public void preprocessCosts(ArrayRealVector[] inputs,
			ArrayRealVector[] desiredOutputs, SplitNetwork network) 
	{
		totalCost=0.0;
		
		for(Entry<State, double[]> stateEntry: ((NeuralMCTSPolicy)policy).stateInfo.entrySet())
		{
			if(stateEntry.getValue()[1]>25)
			{
				StateAction[] possibleStateActions=environment.getAllPossibleStateActions(stateEntry.getKey());
				
				double[] stateValues=new double[possibleStateActions.length];
				double totalStateValue=0.0;
				double[] networkStateValues=new double[possibleStateActions.length];
				double totalNetworkStateValue=0.0;
				
				for(int stateActionInd=0; stateActionInd<possibleStateActions.length; stateActionInd++)
				{
					StateAction possibleStateAction=possibleStateActions[stateActionInd];
					Double value=policy.getStateValueNoNet(possibleStateAction.state);
					if(value!=null)
					{
						value=policy.scaleFilterOutputs.scaleData(new ArrayRealVector(new double[]{value})).getEntry(0);
						double networkValue=network.getOutput(policy.scaleFilterInputs.scaleData(new ArrayRealVector(possibleStateAction.state.getValue()))).getEntry(0);	
						
						value+=policy.rewardFunction.getReward(stateEntry.getKey(), possibleStateAction.state, 
								possibleStateAction.action);
						networkValue+=policy.rewardFunction.getReward(stateEntry.getKey(), possibleStateAction.state, 
								possibleStateAction.action);
						
						stateValues[stateActionInd]=Math.pow(value, exponent);
						totalStateValue+=stateValues[stateActionInd];
						
						networkStateValues[stateActionInd]=Math.pow(networkValue, exponent);
						totalNetworkStateValue+=networkStateValues[stateActionInd];
					}
				}
				
				if(totalStateValue!=0 && totalNetworkStateValue!=0)
				{	
					for(int costInd=0; costInd<possibleStateActions.length; costInd++)
					{
						totalCost+=
								Math.abs(stateValues[costInd]/totalStateValue-networkStateValues[costInd]/totalNetworkStateValue);
					}
				}
			}
		}
	}
	
	private double getDevTotal(double[] stateValues, double[] networkStateValues, int derivativeInd)
	{
		double devTotal=0.0;
		for(int devTotalStateInd=0; devTotalStateInd<stateValues.length; devTotalStateInd++)
		{
			if(derivativeInd==devTotalStateInd)
			{
				devTotal+=3.0*Math.pow(networkStateValues[derivativeInd], 0.666666667);
			}
			else
			{
				devTotal+=networkStateValues[devTotalStateInd];
			}
		}
		devTotal*=networkStateValues[derivativeInd];
		return devTotal;
	}
	
	private double getDerivative(double[] state)
	{
		DoubleWrapper dw=new DoubleWrapper(state);
		Double res=derivatives.get(dw);
		return derivatives.getOrDefault(dw, 0.0);
	}
	
	private void incDerivative(double[] state, double inc)
	{
		DoubleWrapper dw=new DoubleWrapper(state);
		derivatives.put(dw, getDerivative(state)+inc);
	}
	
}

class DoubleWrapper
{
	
	private double[] data;
	
	public DoubleWrapper(double[] data)
	{
		this.data=data;
	}
	
	public boolean equals(Object someOther) 
	{
		return Arrays.equals(data, ((DoubleWrapper)someOther).data);
	}
		   
	public int hashCode() 
	{
		return Arrays.hashCode(data);
	}
		   
}
