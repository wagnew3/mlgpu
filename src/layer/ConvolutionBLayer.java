package layer;

import org.apache.commons.math3.distribution.NormalDistribution;

import activationFunctions.ActivationFunction;
import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;

public class ConvolutionBLayer extends BLayer
{
	
	protected int[] inputsDimensions;
	protected int stride;
	protected int[] boxSize;
	protected int numberRows;
	protected int numberColumns;
	protected Matrix fullWeights;

	public ConvolutionBLayer(ActivationFunction activationFunction, BLayer[] inputLayers, int[] inputsDimensions, int stride, int[] boxSize) 
	{
		super(activationFunction, inputLayers);
		this.inputsDimensions=inputsDimensions;
		this.stride=stride;
		if((inputsDimensions[0]-boxSize[0])%stride!=0)
		{
			System.out.println("inputsDimesions[0]%stride!=boxSize-1");
		}
		if((inputsDimensions[1]-boxSize[1])%stride!=0)
		{
			System.out.println("inputsDimesions[1]%stride!=boxSize-1");
		}
		this.boxSize=boxSize;
		
		weights=new Matrix[inputLayers.length];
		
		for(int inputInd=0; inputInd<inputLayers.length; inputInd++)
		{
			weights[inputInd]=new FDMatrix(1, boxSize[0]*boxSize[1]);
			NormalDistribution nInvGaussian=new NormalDistribution(0.0, 1.0/Math.sqrt(boxSize[0]*boxSize[1]));
			for(int rowIndex=0; rowIndex<weights[inputInd].getRows(); rowIndex++)
			{
				for(int colIndex=0; colIndex<weights[inputInd].getCols(); colIndex++)
				{
					weights[inputInd].set(rowIndex, colIndex, (float)nInvGaussian.sample());
				}
			}
		}
		
		biases=new FDMatrix(1, 1);
		NormalDistribution zeroGuassian=new NormalDistribution(0.0, 1.0);
		biases.set(0, 0, (float)zeroGuassian.sample());
		
		numberRows=(inputsDimensions[0]-Math.max(boxSize[0]-stride, 0))/stride;
		numberColumns=(inputsDimensions[1]-Math.max(boxSize[1]-stride, 0))/stride;
	}

	@Override
	public Matrix getOutput(Matrix[] inputs, BLayer[] inputLayer, Matrix result) 
	{
		Matrix activation=result;
		activation.omatVecMultScaleAdd(weights[0], inputs[0], 1.0f, biases, activation, 0.0f);
		for(int inputInd=0; inputInd<inputs.length; inputInd++)
		{
			activation=activation.omatVecMultScaleAdd(weights[inputInd], inputs[inputInd], 1.0f, biases, activation, 1.0f);
		}
		
		return activationFunction.applyActivationFunction(activation);
	}
	
	@Override
	public int getOutputSize()
	{
		return numberRows*numberColumns;
	}

	@Override
	public ArrayRealVector getActivations(ArrayRealVector input, int weightIndex)
	{
		BlockRealMatrix matrixInputs=new BlockRealMatrix(inputsDimensions[0], inputsDimensions[1]);
		ArrayRealVector z=new ArrayRealVector(getOutputSize());
		for(int rowInd=0; rowInd<matrixInputs.getRowDimension(); rowInd++)
		{
			matrixInputs.setRow(rowInd, input.getSubVector(rowInd*inputsDimensions[1], inputsDimensions[1]).toArray());
		}
		
		int neuronIndex=0;
		for(int rowInd=0; rowInd<numberRows; rowInd+=stride)
		{
			for(int colInd=0; colInd<numberColumns; colInd+=stride)
			{
				z.setEntry(neuronIndex, vectorMatrixDotProduct(weights[weightIndex], matrixInputs.getSubMatrix(rowInd, rowInd+boxSize[0]-1, colInd, colInd+boxSize[1]-1)));
				neuronIndex++;
			}
		}
		z=(ArrayRealVector)z.mapAdd(biases.getEntry(0));
		return z;
	}

	@Override
	public Matrix getOutputDerivatives(Matrix[] inputs, Matrix result) 
	{
		Matrix activation=result;
		activation.omatVecMultScaleAdd(weights[0], inputs[0], 1.0f, biases, activation, 0.0f);
		for(int inputInd=0; inputInd<inputs.length; inputInd++)
		{
			activation=activation.omatVecMultScaleAdd(weights[inputInd], inputs[inputInd], 1.0f, biases, activation, 1.0f);
		}
		return activationFunction.getDerivatives(activation);
	}

	@Override
	public ArrayRealVector getDeltas(BlockRealMatrix[] nextLayerWeights, ArrayRealVector[] nextLayerDeltas, ArrayRealVector activationDerivative) 
	{
		ArrayRealVector weightsDeltasSum=new ArrayRealVector(getOutputSize());
		for(int outputInd=0; outputInd<nextLayerWeights.length; outputInd++)
		{
			weightsDeltasSum=weightsDeltasSum.add(nextLayerWeights[outputInd].transpose().operate(nextLayerDeltas[outputInd]));
		}
		
		return (ArrayRealVector)weightsDeltasSum.ebeMultiply(activationDerivative);
	}
	
	@Override
	public RealMatrix getWeightPDs(ArrayRealVector previousLayerOutputs, ArrayRealVector deltas) 
	{
		double[][] weights=fullWeights.getData();
		double[][] weightsPD=deltas.outerProduct(previousLayerOutputs).getData();
		
		for(int rowInd=0; rowInd<weights.length; rowInd++)
		{
			for(int colInd=0; colInd<weights[rowInd].length; colInd++)
			{
				if(weights[rowInd][colInd]==0.0)
				{
					weightsPD[rowInd][colInd]=0.0;
				}
			}
		}
		
		return new BlockRealMatrix(weightsPD);
	}
	
	public BlockRealMatrix[] getWeights()
	{
		BlockRealMatrix[] fullWeights=new BlockRealMatrix[inputLayers.length];
		
		for(int inputInd=0; inputInd<inputLayers.length; inputInd++)
		{
			fullWeights[inputInd]=new BlockRealMatrix(getOutputSize(), inputsDimensions[0]*inputsDimensions[1]);
			for(int rowInd=0; rowInd<numberRows; rowInd+=stride)
			{
				for(int colInd=0; colInd<numberColumns; colInd+=stride) //for each neuron
				{
					
					for(int boxRowInd=0; boxRowInd<boxSize[0]; boxRowInd++) //for each input in neuron's box
					{
						for(int boxColInd=0; boxColInd<boxSize[1]; boxColInd++)
						{
							int neuronIndex=(rowInd/stride)*numberColumns+(colInd/stride);
							fullWeights[inputInd].setEntry(neuronIndex, (rowInd+boxRowInd)*numberColumns+(colInd+boxColInd), weights[inputInd].getEntry(0, boxRowInd*boxSize[0]+boxColInd));
						}
					}
					
				}
			}
		}
		
		return fullWeights;
	}
	
	public BlockRealMatrix getFullWeights()
	{
		BlockRealMatrix fullWeights=new BlockRealMatrix(getOutputSize(), inputsDimensions[0]*inputsDimensions[1]);
		
		for(int rowInd=0; rowInd<numberRows; rowInd+=stride)
		{
			for(int colInd=0; colInd<numberColumns; colInd+=stride) //for each neuron
			{
				
				for(int boxRowInd=0; boxRowInd<boxSize[0]; boxRowInd++) //for each input in neuron's box
				{
					for(int boxColInd=0; boxColInd<boxSize[1]; boxColInd++)
					{
						int neuronIndex=(rowInd/stride)*numberColumns+(colInd/stride);
						fullWeights.setEntry(neuronIndex, (rowInd+boxRowInd)*numberColumns+(colInd+boxColInd), weights[0].getEntry(0, boxRowInd*boxSize[0]+boxColInd));
					}
				}
				
			}
		}
		
		return fullWeights;
	}
	
	public ArrayRealVector getBiases()
	{
		ArrayRealVector fullBiases=new ArrayRealVector(getOutputSize(), biases.getEntry(0));
		return fullBiases;
	}
	
	//weightpds=deltas*activations
	// sum and average columns to get weight pds with single delta (single bias derivative)
	//result is inputs, add to appropriate weight variable for weights pd 
	public void updateWeights(BlockRealMatrix[] weightPDs, double learningRate)
	{
		for(int inputInd=0; inputInd<inputLayers.length; inputInd++)
		{
			ArrayRealVector weightPDsVector=new ArrayRealVector(weightPDs[inputInd].getColumnDimension());
			ArrayRealVector columnAdd=new ArrayRealVector(weightPDs[inputInd].getRowDimension(), 1.0);
			for(int columnInd=0; columnInd<weightPDsVector.getDimension(); columnInd++)
			{
				ArrayRealVector column=(ArrayRealVector)weightPDs[inputInd].getColumnVector(columnInd);
				weightPDsVector.setEntry(columnInd, columnAdd.dotProduct(column));
			}
			
			BlockRealMatrix condensedWeightsPDVector=new BlockRealMatrix(1, weights[inputInd].getColumnDimension());
			for(int columnInd=0; columnInd<weightPDsVector.getDimension(); columnInd++)
			{
				int row=columnInd/inputsDimensions[0];
				int column=columnInd%inputsDimensions[0];
				
				condensedWeightsPDVector.addToEntry(0, (row%boxSize[0])*boxSize[0]+column%boxSize[1], weightPDsVector.getEntry(columnInd));
			}
			
			weights[inputInd]=weights[inputInd].subtract(condensedWeightsPDVector.scalarMultiply(learningRate));
		}
		fullWeights=getFullWeights();
	}
	
	public void updateBiases(ArrayRealVector biasesPDs, double learningRate)
	{
		if(Double.isNaN(biasesPDs.getEntry(0)))
		{
			int i=0;
		}
		double biasPD=0.0;
		for(int biasInd=0; biasInd<biasesPDs.getDimension(); biasInd++)
		{
			biasPD+=biasesPDs.getEntry(biasInd);
		}
		biasPD/=biasesPDs.getDimension();
		biases=(ArrayRealVector)biases.mapSubtract(biasPD*learningRate);
	}

	@Override
	public Layer clone() 
	{
		// TODO Auto-generated method stub
		return null;
	}

}
