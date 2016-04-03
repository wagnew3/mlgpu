package utils;

public class UnbalancedBitBinTree<K, V>
{
    
    private final int numberStateBits;
    
    BinTreeNode head;
    
    public UnbalancedBitBinTree(int numberStateBits)
    {
	this.numberStateBits=numberStateBits;
	head=new BinTreeNode();
    }
    
    public void add(K key, V value)
    {
	for(int i)
    }
    
    public V get(K key)
    {
	
    }
    
    public void remove(K key)
    {
	
    }

}

class BinTreeNode
{
    boolean leaf;
    Object left;
    Object right;
}
