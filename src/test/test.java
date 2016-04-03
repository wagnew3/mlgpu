package test;

import org.apache.commons.math3.linear.ArrayRealVector;

import activationFunctions.L2Pooling;
import activationFunctions.Sigmoid;
import costFunctions.EuclideanDistanceCostFunction;
import network.Network;

public class test 
{
	
	public static void main(String[] args)
	{
		//testSingleWidth();
		//0.715042105700989+0.723121805124389+0.731058578630004+0.738850006084248
		System.out.println(-0.00036398810582964*0.6);
		
		/*
		double activation=0.546870223010856;
		
		System.out.println(new EuclideanDistanceCostFunction().getCost(new ArrayRealVector(new double[]{activation}), new ArrayRealVector(new double[]{0.99})));
		System.out.println(new EuclideanDistanceCostFunction().getCostDerivative(new ArrayRealVector(new double[]{activation}), new ArrayRealVector(new double[]{0.99})));
		*/
		
		/*
		System.out.println(activation);

		System.out.println(new Sigmoid().applyActivationFunction(activation+0.05));
		System.out.println(new Sigmoid().getDerivative(activation+0.05));
		*/
		
		/*
		System.out.println(new L2Pooling().applyPoolingActivationFunction(new ArrayRealVector(new double[]{activation})));
		System.out.println(new L2Pooling().getPoolingDerivatives(new ArrayRealVector(new double[]{activation})));
		*/
		
	}
	
	public static void testSingleWidth()
	{
		Network network=new FCLFFNetwork(new int[]{2, 2, 2}, new Sigmoid(), new BackpropagationForwardLayers());
		
		double[][] inputs=new double[][]{new double[]{0.05, 0.1}};
		double[][] outputs=new double[][]{new double[]{0.01, 0.99}};
		
		FCLFFNetwork fNet=(FCLFFNetwork)network;
		
		fNet.getLayer(1)[0].setWeight(.15, 0);
		fNet.getLayer(1)[0].setWeight(.20, 1);
		fNet.getLayer(1)[0].setBias(.35);
		
		fNet.getLayer(1)[1].setWeight(.25, 0);
		fNet.getLayer(1)[1].setWeight(.30, 1);
		fNet.getLayer(1)[1].setBias(.35);
		
		
		fNet.getLayer(2)[0].setWeight(.40, 0);
		fNet.getLayer(2)[0].setWeight(.45, 1);
		fNet.getLayer(2)[0].setBias(.60);
		
		fNet.getLayer(2)[1].setWeight(.50, 0);
		fNet.getLayer(2)[1].setWeight(.55, 1);
		fNet.getLayer(2)[1].setBias(.60);
		
		network.learn(inputs, outputs, new EuclideanErrorFunction());
	}

}
