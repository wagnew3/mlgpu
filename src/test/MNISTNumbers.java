package test;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

import activationFunctions.ActivationFunction;
import activationFunctions.L2Pooling;
import activationFunctions.PoolingActivationFunction;
import activationFunctions.RectifiedLinearActivationFunction;
import activationFunctions.Sigmoid;
import activationFunctions.SoftMax;
import costFunctions.CrossEntropy;
import costFunctions.EuclideanDistanceCostFunction;
import filters.ScaleFilter;
import jcuda.jcublas.JCublas2;
import jcuda.runtime.JCuda;
import layer.BInputLayer;
import layer.BLayer;
import layer.ConvolutionBLayer;
import layer.ConvolutionLayer;
import layer.FullyConnectedBLayer;
import layer.FullyConnectedLayer;
import layer.InputLayer;
import layer.Layer;
import layer.PoolingLayer;
import learningRule.BPGDUnsupervisedTraining;
import learningRule.BackPropGradientDescent;
import learningRule.BackPropGradientDescentMultiThreaded;
import learningRule.MPBackPropGradientDescent;
import learningRule.RProp;
import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;
import network.Network;
import network.SplitFeedForwardNetwork;
import network.SplitNetwork;
import regularization.L2Regularization;
import network.FeedForwardNetwork;


public class MNISTNumbers 
{
	
	static Matrix[] evalImages;
	static double[][] evalLabels;
	static ScaleFilter scaleFilter=null;
	
	public static void main(String[] args) throws IOException
	{
		JCuda.setExceptionsEnabled(true);
        JCublas2.setExceptionsEnabled(true);
        
		//verify();
		//evalNNBasicBP();
		//evalNNBasicBPMT();
		//evalConvNNBasicBP();
		//validateConvNNBasicBP();
		//evalNNSplitBasic();
		evalNNSplitBasicRProp();
		//evalNNSplitBasicPretrain();
		//evalNNSplitSplitInput();
		//evalNNSplitSplitMiddle();
		//evalNNSplitConvo();
		//verifyNNSplitBasic();
	}
	
	public static void verify() //http://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
	{
		InputLayer inputLayer=new InputLayer(null, 2);
		FullyConnectedLayer hiddenLayer=new FullyConnectedLayer(new Sigmoid(), inputLayer, 2);
		hiddenLayer.setBiases(new ArrayRealVector(new double[]{0.35, 0.35}));
		hiddenLayer.setWeights(new BlockRealMatrix(new double[][]{new double[]{0.15, 0.20}, new double[]{0.25, 0.30}}));
		FullyConnectedLayer outputLayer=new FullyConnectedLayer(new Sigmoid(), hiddenLayer, 2);
		outputLayer.setBiases(new ArrayRealVector(new double[]{0.60, 0.60}));
		outputLayer.setWeights(new BlockRealMatrix(new double[][]{new double[]{0.40, 0.45}, new double[]{0.50, 0.55}}));
		Network network=new FeedForwardNetwork(new Layer[]{inputLayer, hiddenLayer, outputLayer});
		
		double[][] trainImages=new double[][]{new double[]{0.05, 0.1}};
		double[][] trainLabels=new double[][]{new double[]{0.01, 0.99}};
		new BackPropGradientDescent(10, 30, 0.5).trainNetwork(network, doublessToArrays(trainImages),
				doublessToArrays(trainLabels), new CrossEntropy());
	}
	
	public static void evalNNSplitBasic() throws IOException
	{
		Object[] training=getImagesAndLabels("train");
		double[][] trainImages=(double[][])training[0];
		double[][] trainLabels=(double[][])training[1];
		
		/*
		double[][] reduTrainImages=new double[5000][];
		double[][] reduTrainLabels=new double[5000][];
		for(int imageInd=0; imageInd<reduTrainImages.length; imageInd++)
		{
			reduTrainImages[imageInd]=trainImages[imageInd];
			reduTrainLabels[imageInd]=trainLabels[imageInd];
		}
		*/
		
		scaleFilter=new ScaleFilter();
				
		Object[] eval=getImagesAndLabels("t10k");
		evalImages=doublessToArrays((double[][])eval[0]);
		evalLabels=(double[][])eval[1];
		
		/*
		InputLayer inputLayer=new InputLayer(null, 28*28);
		FullyConnectedLayer hiddenLayer1=new FullyConnectedLayer(new Sigmoid(), inputLayer, 250);
		FullyConnectedLayer hiddenLayer2=new FullyConnectedLayer(new Sigmoid(), hiddenLayer1, 50);
		FullyConnectedLayer outputLayer=new FullyConnectedLayer(new Sigmoid(), hiddenLayer2, 10);
		Network network=new FeedForwardNetwork(new Layer[]{inputLayer, hiddenLayer1, hiddenLayer2, outputLayer});
		*/
		
		BInputLayer inputLayer=new BInputLayer(null, null, 28*28);
		FullyConnectedBLayer hiddenLayer1=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{inputLayer}, 25);
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer1}, 10);
		SplitNetwork network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer}, new BLayer[]{hiddenLayer1}, new BLayer[]{outputLayer}});
		
		float lambda=0.5f;
		
		MPBackPropGradientDescent bpgd=new MPBackPropGradientDescent(1, 50, lambda);
		//bpgd.setRegularization(new L2Regularization(outputLayer.getOutputSize(), lambda, 0.1));
		new MPBackPropGradientDescent(100, 50, lambda).trainNetwork(network, doublessToArrayss(trainImages),
				doublessToArrayss(trainLabels), new EuclideanDistanceCostFunction());
	}
	
	public static void evalNNSplitBasicRProp() throws IOException
	{
		Object[] training=getImagesAndLabels("train");
		double[][] trainImages=(double[][])training[0];
		double[][] trainLabels=(double[][])training[1];
		
		scaleFilter=new ScaleFilter();
				
		Object[] eval=getImagesAndLabels("t10k");
		evalImages=doublessToArrays((double[][])eval[0]);
		evalLabels=(double[][])eval[1];

		BInputLayer inputLayer=new BInputLayer(null, null, 28*28);
		FullyConnectedBLayer hiddenLayer1=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{inputLayer}, 25);
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer1}, 10);
		SplitNetwork network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer}, new BLayer[]{hiddenLayer1}, new BLayer[]{outputLayer}});
		
		long time=System.nanoTime();
		new RProp(100, 50, 0.1f).trainNetwork(network, doublessToArrayss(trainImages),
				doublessToArrayss(trainLabels), new EuclideanDistanceCostFunction());
		time=System.nanoTime()-time;
		System.out.println(time);
	}
	
	public static void evalNNSplitBasicPretrain() throws IOException
	{
		Object[] training=getImagesAndLabels("train");
		double[][] trainImages=(double[][])training[0];
		double[][] trainLabels=(double[][])training[1];
		
		scaleFilter=new ScaleFilter();
				
		Object[] eval=getImagesAndLabels("t10k");
		evalImages=doublessToArrays((double[][])eval[0]);
		evalLabels=(double[][])eval[1];
		/*
		InputLayer inputLayer=new InputLayer(null, 28*28);
		FullyConnectedLayer hiddenLayer1=new FullyConnectedLayer(new Sigmoid(), inputLayer, 250);
		FullyConnectedLayer hiddenLayer2=new FullyConnectedLayer(new Sigmoid(), hiddenLayer1, 50);
		FullyConnectedLayer outputLayer=new FullyConnectedLayer(new Sigmoid(), hiddenLayer2, 10);
		Network network=new FeedForwardNetwork(new Layer[]{inputLayer, hiddenLayer1, hiddenLayer2, outputLayer});
		*/
		
		BInputLayer inputLayer=new BInputLayer(null, null, 28*28);
		ConvolutionBLayer hiddenLayer1=new ConvolutionBLayer(new Sigmoid(), new BLayer[]{inputLayer}, new int[]{28, 28}, 1, 5);
		FullyConnectedBLayer hiddenLayer2=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer1}, 100);
		FullyConnectedBLayer hiddenLayer3=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer2}, 20);
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer3}, 10);
		SplitNetwork network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer}, new BLayer[]{hiddenLayer1}, new BLayer[]{hiddenLayer2}, new BLayer[]{hiddenLayer3}, new BLayer[]{outputLayer}});
		
		double lambda=0.1;
		
		BPGDUnsupervisedTraining bpgd=new BPGDUnsupervisedTraining(10, 30, lambda, null);
		//bpgd.setRegularization(new L2Regularization(outputLayer.getOutputSize(), lambda, 0.1));
		
		bpgd.unsupervisedTrain(network, doublessToArrays(trainImages),
		doublessToArrays(trainLabels), new EuclideanDistanceCostFunction());
		
		bpgd.trainNetwork(network, doublessToArrays(trainImages),
				doublessToArrays(trainLabels), new EuclideanDistanceCostFunction());
	}
	
	public static void evalNNSplitSplitInput() throws IOException
	{
		Object[] training=getImagesAndLabels("train");
		double[][] trainImages=(double[][])training[0];
		double[][] trainLabels=(double[][])training[1];
		
		Object[] eval=getImagesAndLabels("t10k");
		evalImages=doublessToArrays((double[][])eval[0]);
		evalLabels=(double[][])eval[1];
		
		/*
		InputLayer inputLayer=new InputLayer(null, 28*28);
		FullyConnectedLayer hiddenLayer1=new FullyConnectedLayer(new Sigmoid(), inputLayer, 250);
		FullyConnectedLayer hiddenLayer2=new FullyConnectedLayer(new Sigmoid(), hiddenLayer1, 50);
		FullyConnectedLayer outputLayer=new FullyConnectedLayer(new Sigmoid(), hiddenLayer2, 10);
		Network network=new FeedForwardNetwork(new Layer[]{inputLayer, hiddenLayer1, hiddenLayer2, outputLayer});
		*/
		
		BInputLayer inputLayer1=new BInputLayer(null, null, 28*28);
		BInputLayer inputLayer2=new BInputLayer(null, null, 28*28);
		FullyConnectedBLayer hiddenLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{inputLayer1, inputLayer2}, 20);
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer}, 10);
		SplitNetwork network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer1, inputLayer2}, new BLayer[]{hiddenLayer}, new BLayer[]{outputLayer}});
		
		new MPBackPropGradientDescent(10, 30, 0.1).trainNetwork(network, doublessToArrays(trainImages),
				doublessToArrays(trainLabels), new CrossEntropy());
	}
	
	public static void evalNNSplitSplitMiddle() throws IOException
	{
		Object[] training=getImagesAndLabels("train");
		double[][] trainImages=(double[][])training[0];
		double[][] trainLabels=(double[][])training[1];
		
		Object[] eval=getImagesAndLabels("t10k");
		evalImages=doublessToArrays((double[][])eval[0]);
		evalLabels=(double[][])eval[1];
		
		BInputLayer inputLayer1=new BInputLayer(null, null, 28*28);
		FullyConnectedBLayer hiddenLayer1=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, 20);
		FullyConnectedBLayer hiddenLayer2=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, 20);
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer1, hiddenLayer2}, 10);
		SplitNetwork network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer1}, new BLayer[]{hiddenLayer1, hiddenLayer2}, new BLayer[]{outputLayer}});
		
		new MPBackPropGradientDescent(10, 30, 0.1).trainNetwork(network, doublessToArrays(trainImages),
				doublessToArrays(trainLabels), new CrossEntropy());
	}
	
	public static void evalNNSplitConvo() throws IOException
	{
		Object[] training=getImagesAndLabels("train");
		double[][] trainImages=(double[][])training[0];
		double[][] trainLabels=(double[][])training[1];
		
		Object[] eval=getImagesAndLabels("t10k");
		evalImages=doublessToArrays((double[][])eval[0]);
		evalLabels=(double[][])eval[1];
		
		/*
		InputLayer inputLayer=new InputLayer(null, 28*28);
		FullyConnectedLayer hiddenLayer1=new FullyConnectedLayer(new Sigmoid(), inputLayer, 250);
		FullyConnectedLayer hiddenLayer2=new FullyConnectedLayer(new Sigmoid(), hiddenLayer1, 50);
		FullyConnectedLayer outputLayer=new FullyConnectedLayer(new Sigmoid(), hiddenLayer2, 10);
		Network network=new FeedForwardNetwork(new Layer[]{inputLayer, hiddenLayer1, hiddenLayer2, outputLayer});
		*/
		
		BInputLayer inputLayer1=new BInputLayer(null, null, 28*28);
		ConvolutionBLayer hiddenLayer1=new ConvolutionBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, new int[]{28, 28}, 1, 5);
		//FullyConnectedBLayer hiddenLayer2=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{inputLayer1}, 20);
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer1}, 10);
		SplitNetwork network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer1}, new BLayer[]{hiddenLayer1}, new BLayer[]{outputLayer}});
		
		new MPBackPropGradientDescent(10, 30, 0.1).trainNetwork(network, doublessToArrays(trainImages),
				doublessToArrays(trainLabels), new CrossEntropy());
	}
	
	public static void verifyNNSplitBasic() //http://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
	{
		BInputLayer inputLayer=new BInputLayer(null, null, 2);
		FullyConnectedBLayer hiddenLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{inputLayer}, 2);
		hiddenLayer.setBiases(new FDMatrix(new float[][]{new float[]{0.35f}, new float[]{0.35f}}));
		hiddenLayer.setWeights(new FDMatrix[]{new FDMatrix(new float[][]{new float[]{0.15f, 0.20f}, new float[]{0.25f, 0.30f}})});
		FullyConnectedBLayer outputLayer=new FullyConnectedBLayer(new Sigmoid(), new BLayer[]{hiddenLayer}, 2);
		outputLayer.setBiases(new FDMatrix(new float[][]{new float[]{0.60f}, new float[]{ 0.60f}}));
		outputLayer.setWeights(new FDMatrix[]{new FDMatrix(new float[][]{new float[]{0.40f, 0.45f}, new float[]{0.50f, 0.55f}})});
		SplitNetwork network=new SplitFeedForwardNetwork(new BLayer[][]{new BLayer[]{inputLayer}, new BLayer[]{hiddenLayer}, new BLayer[]{outputLayer}});
		
		
		double[][] trainImages=new double[][]{new double[]{0.05f, 0.1f}};
		double[][] trainLabels=new double[][]{new double[]{0.01f, 0.99f}};
		new MPBackPropGradientDescent(10, 30, 0.5f).trainNetwork(network, doublessToArrays(trainImages),
				doublessToArrays(trainLabels), new EuclideanDistanceCostFunction());
	}
	
	public static void evalNumberNet(Network network)
	{
		int numberErrors=0;
		for(int evalInd=0; evalInd<evalImages.length; evalInd++)
		{
			float[] output=network.getOutput(evalImages[evalInd]);
			
			int number=-1;
			double max=Double.NEGATIVE_INFINITY;
			for(int outputInd=0; outputInd<output.length; outputInd++)
			{
				if(output[outputInd]>max)
				{
					number=outputInd;
					max=output[outputInd];
				}
			}
			
			try
			{
			if(evalLabels[evalInd][number]==0)
			{
				numberErrors++;
			}
			}
			catch(Exception e)
			{
				e.printStackTrace();
			}
		}
		
		System.out.println(numberErrors+" incorrect out of "+evalImages.length);
	}
	
	public static void evalNumberNet(SplitNetwork network)
	{
		int numberErrors=0;
		for(int evalInd=0; evalInd<evalImages.length; evalInd++)
		{
			float[] output=network.getOutput(new Matrix[]{evalImages[evalInd]})[0].getData();
			
			int number=-1;
			double max=Double.NEGATIVE_INFINITY;
			for(int outputInd=0; outputInd<output.length; outputInd++)
			{
				if(output[outputInd]>max)
				{
					number=outputInd;
					max=output[outputInd];
				}
			}
			
			try
			{
			if(evalLabels[evalInd][number]==0)
			{
				numberErrors++;
			}
			}
			catch(Exception e)
			{
				e.printStackTrace();
			}
		}
		
		System.out.println(numberErrors+" incorrect out of "+evalImages.length);
	}
	
	private static Object[] getImagesAndLabels(String fileName) throws IOException
	{
	    BufferedInputStream labels = new BufferedInputStream(new FileInputStream("/home/willie/workspace/mlGPU/data/MNIST_Numbers/"+fileName+"-labels.idx1-ubyte"));
		BufferedInputStream images = new BufferedInputStream(new FileInputStream("/home/willie/workspace/mlGPU/data/MNIST_Numbers/"+fileName+"-images.idx3-ubyte"));
	    
		byte[] intBytes=new byte[4];
		labels.read(intBytes);
		int magicNumber = b8ToB10Int(intBytes);
	    if (magicNumber != 2049) {
	      System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
	      System.exit(0);
	    }
	    images.read(intBytes);
	    magicNumber = b8ToB10Int(intBytes);
	    if (magicNumber != 2051) {
	      System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
	      System.exit(0);
	    }
	    labels.read(intBytes);
	    int numLabels = b8ToB10Int(intBytes);
	    images.read(intBytes);
	    int numImages = b8ToB10Int(intBytes);
	    images.read(intBytes);
	    int numRows = b8ToB10Int(intBytes);
	    images.read(intBytes);
	    int numCols = b8ToB10Int(intBytes);
	    if (numLabels != numImages) {
	      System.err.println("Image file and label file do not contain the same number of entries.");
	      System.err.println("  Label file contains: " + numLabels);
	      System.err.println("  Image file contains: " + numImages);
	      System.exit(0);
	    }

	    long start = System.currentTimeMillis();
	    int numLabelsRead = 0;
	    int numImagesRead = 0;
	    double[][] imagesArray=new double[numImages][28*28];
	    double[][] labelsArray=new double[numImages][10];
	    byte[] labelsBytes=new byte[numImages];
	    labels.read(labelsBytes);
	    byte[] imageBytes=new byte[numImages*28*28];
	    images.read(imageBytes);
	    while (numLabelsRead < numLabels) 
	    {
	    	labelsArray[numLabelsRead][labelsBytes[numLabelsRead]]=1.0;
	    	numLabelsRead++;
	      for (int colIdx = 0; colIdx < numCols; colIdx++) 
	      {
	        for (int rowIdx = 0; rowIdx < numRows; rowIdx++) 
	        {
	        	imagesArray[numImagesRead][28*colIdx+rowIdx]=(double)(Byte.toUnsignedInt(imageBytes[28*28*numImagesRead+28*colIdx+rowIdx]))/256;
	        }
	      }
	      double[] image=imagesArray[0];
	      numImagesRead++;
	    }
	    
	    long end = System.currentTimeMillis();
	    long elapsed = end - start;
	    System.out.println("Read " + numLabelsRead + " samples in " + elapsed + " ms ");
	    return new Object[]{imagesArray, labelsArray};
		
	}
	
	public static int b8ToB10Int(byte[] b8)
    {
        return ((b8[0] & 0xFF) << 24) |
         ((b8[1] & 0xFF) << 16) |
         ((b8[2] & 0xFF) <<  8) |
         (b8[3] & 0xFF);
    }
	
	public static Matrix[] doublessToArrays(double[][] doubless)
	{
		Matrix[] arrays=new Matrix[doubless.length];
		for(int doublesInd=0; doublesInd<doubless.length; doublesInd++)
		{
			float[] floats=new float[doubless[doublesInd].length];
			for(int doublessInd=0; doublessInd<floats.length; doublessInd++)
			{
				floats[doublessInd]=(float)doubless[doublesInd][doublessInd];
			}
			arrays[doublesInd]=new FDMatrix(floats, doubless[doublesInd].length, 1);
		}
		return arrays;
	}
	
	public static Matrix[][] doublessToArrayss(double[][] doubless)
	{
		Matrix[][] arrayss=new Matrix[doubless.length][];
		for(int doublesInd=0; doublesInd<doubless.length; doublesInd++)
		{
			float[] floats=new float[doubless[doublesInd].length];
			for(int doublessInd=0; doublessInd<floats.length; doublessInd++)
			{
				floats[doublessInd]=(float)doubless[doublesInd][doublessInd];
			}
			arrayss[doublesInd]=new Matrix[]{new FDMatrix(floats, doubless[doublesInd].length, 1)};
		}
		return arrayss;
	}

}
