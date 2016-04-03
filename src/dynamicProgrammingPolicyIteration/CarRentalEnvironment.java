package dynamicProgrammingPolicyIteration;

import org.apache.commons.math3.distribution.PoissonDistribution;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.Environment;
import ReinforcementMachineLearningFramework.ActionPolicy;
import ReinforcementMachineLearningFramework.State;

public class CarRentalEnvironment extends Environment
{
	private static final PoissonDistribution location1Rental=new PoissonDistribution(3.0);
	private static final PoissonDistribution location2Rental=new PoissonDistribution(4.0);
	private static final PoissonDistribution location1Return=new PoissonDistribution(3.0);
	private static final PoissonDistribution location2Return=new PoissonDistribution(2.0);
	private static final int maxCars=20;
	
	private CarRentalState currentState;
	private CarRentalState startState;
	
	public CarRentalEnvironment()
	{
	}
	
	@Override
	public State getCurrentState() 
	{
		return currentState;
	}

	@Override
	public void step(ActionPolicy policy) 
	{
		CarRentalState movedState=(CarRentalState)currentState.getSuccessor(policy.getAction(currentState));
		int carsRentedLocation1=(int)Math.round(location1Rental.sample());
		int carsRentedLocation2=(int)Math.round(location2Rental.sample());
		int carsReturnedLocation1=(int)Math.round(location1Return.sample());
		int carsReturnedLocation2=(int)Math.round(location2Return.sample());
		
		int carsActuallyLocation1=Math.min(Math.max(movedState.numberCarsLocation1-carsRentedLocation1+currentState.numberCarsReturnedLocation1, 0), 20);
		int carsActuallyRentedLocation1=Math.min(movedState.numberCarsLocation1+currentState.numberCarsReturnedLocation1, carsRentedLocation1);
		
		int carsActuallyLocation2=Math.min(Math.max(movedState.numberCarsLocation2-carsRentedLocation2+currentState.numberCarsReturnedLocation2, 0), 20);
		int carsActuallyRentedLocation2=Math.min(movedState.numberCarsLocation2+currentState.numberCarsReturnedLocation2, carsRentedLocation2);
		
		CarRentalState eveningState=new CarRentalState(carsActuallyLocation1, carsActuallyLocation2, currentState);
		eveningState.numberCarsReturnedLocation1=carsReturnedLocation1;
		eveningState.numberCarsReturnedLocation2=carsReturnedLocation2;
		
		currentState=eveningState;
	}

	@Override
	public void reset() 
	{
		currentState=startState;
	}

	@Override
	public void setStartState(State startState) 
	{
		CarRentalState carRentalStateStartState=(CarRentalState)startState;
		this.startState=carRentalStateStartState;
		currentState=carRentalStateStartState;
	}

	@Override
	public double transistionProbability(State oldState, State newState, Action action) 
	{
		CarRentalState oldCarRentalState=(CarRentalState)oldState;
		CarRentalState newCarRentalState=(CarRentalState)newState;
		CarRentalAction carRentalAction=(CarRentalAction)action;
		
		CarRentalState movedState=(CarRentalState)oldCarRentalState.getSuccessor(carRentalAction);
		
		double chanceLocation1=poissonDifference(location1Return,
				location1Rental,
				newCarRentalState.numberCarsLocation1-movedState.numberCarsLocation1);
		
		double chanceLocation2=poissonDifference(location2Return,
				location2Rental,
				newCarRentalState.numberCarsLocation2-movedState.numberCarsLocation2);
		
		return chanceLocation1*chanceLocation2;
	}
	
	private double poissonDifference(PoissonDistribution poissonDistributionA, PoissonDistribution poissonDistributionB, int difference)
	{
		double probability=0.0;
		for(int poisAValue=0; poisAValue<25; poisAValue++)
		{
			for(int poisBValue=0; poisBValue<25; poisBValue++)
			{
				if(poisAValue-poisBValue==difference)
				{
					probability+=poissonDistributionA.probability(poisAValue)*poissonDistributionC.probability(poisBValue);				
				}		
			}
		}
		return probability;
	}

}
