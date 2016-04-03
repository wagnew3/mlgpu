package utils;

//import RT.EncodeUtils.EndianReader;

public final class UnsecureHash 
{

  private UnsecureHash() {}

  //MurmurHash 3
  public static int hash(byte[] data, int seed) 
  {

    int c1 = 0xcc9e2d51;
    int c2 = 0x1b873593;
    int len=data.length;
    int h1 = seed;
    int roundedEnd = (len & 0xfffffffc); 

    for (int i = 0; i < roundedEnd; i += 4) {

      int k1 = (data[i] & 0xff) | ((data[i + 1] & 0xff) << 8) | ((data[i + 2] & 0xff) << 16) | (data[i + 3] << 24);
      k1 *= c1;
      k1 = (k1 << 15) | (k1 >>> 17);  
      k1 *= c2;

      h1 ^= k1;
      h1 = (h1 << 13) | (h1 >>> 19);  
      h1 = h1 * 5 + 0xe6546b64;
    }


    int k1 = 0;

    switch(len & 0x03) {
      case 3:
        k1 = (data[roundedEnd + 2] & 0xff) << 16;

      case 2:
        k1 |= (data[roundedEnd + 1] & 0xff) << 8;

      case 1:
        k1 |= data[roundedEnd] & 0xff;
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >>> 17);  
        k1 *= c2;
        h1 ^= k1;
      default:
    }


    h1 ^= len;


    h1 ^= h1 >>> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >>> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >>> 16;

    return h1;
  }
  
    public static int hashString(String toHash, int seed) 
    {
        byte[] data=toHash.getBytes();
        int c1 = 0xcc9e2d51;
        int c2 = 0x1b873593;
        int len=data.length;
        int h1 = seed;
        int roundedEnd = (len & 0xfffffffc); 

        for (int i = 0; i < roundedEnd; i += 4) {

          int k1 = (data[i] & 0xff) | ((data[i + 1] & 0xff) << 8) | ((data[i + 2] & 0xff) << 16) | (data[i + 3] << 24);
          k1 *= c1;
          k1 = (k1 << 15) | (k1 >>> 17);  
          k1 *= c2;

          h1 ^= k1;
          h1 = (h1 << 13) | (h1 >>> 19);  
          h1 = h1 * 5 + 0xe6546b64;
        }


        int k1 = 0;

        switch(len & 0x03) {
          case 3:
            k1 = (data[roundedEnd + 2] & 0xff) << 16;

          case 2:
            k1 |= (data[roundedEnd + 1] & 0xff) << 8;

          case 1:
            k1 |= data[roundedEnd] & 0xff;
            k1 *= c1;
            k1 = (k1 << 15) | (k1 >>> 17);  
            k1 *= c2;
            h1 ^= k1;
          default:
        }


        h1 ^= len;


        h1 ^= h1 >>> 16;
        h1 *= 0x85ebca6b;
        h1 ^= h1 >>> 13;
        h1 *= 0xc2b2ae35;
        h1 ^= h1 >>> 16;

        return h1;
  }
  
  public static int hashL(long data, int seed) 
  {

    int c1 = 0xcc9e2d51;
    int c2 = 0x1b873593;
    int len=8;
    int h1 = seed;

    //for (int i = 0; i < roundedEnd; i += 4) {

      long k1 = ((data) & 0xff) | ((data & 0xff) << 8) | ((data & 0xff) << 16) | (data << 24);
      k1 *= c1;
      k1 = (k1 << 15) | (k1 >>> 17);  
      k1 *= c2;
      
      h1 ^= k1;
      h1 = (h1 << 13) | (h1 >>> 19);  
      h1 = h1 * 5 + 0xe6546b64;
      data/=100;
      k1 = (data & 0xff) | ((data & 0xff) << 8) | ((data & 0xff) << 16) | (data << 24);
      k1 *= c1;
      k1 = (k1 << 15) | (k1 >>> 17);  
      k1 *= c2;
     
      h1 ^= k1;
      h1 = (h1 << 13) | (h1 >>> 19);  
      h1 = h1 * 5 + 0xe6546b64;

      k1 = (data/100 & 0xff);
      k1 *= c1;
      k1 = (k1 << 15) | (k1 >>> 17);  
      k1 *= c2;
 
      h1 ^= k1;
      h1 = (h1 << 13) | (h1 >>> 19);  
      h1 = h1 * 5 + 0xe6546b64;
        

      k1 = 0;

    /*switch(len & 0x03) 
    {
      case 3:
        k1 = (data & 0xff) << 16;

      case 2:       
        k1 |= (data & 0xff) << 8;

      case 1:
        k1 |= data & 0xff;
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >>> 17);  
        k1 *= c2;
        h1 ^= k1;
      default:
    }*/


    h1 ^= len;


    h1 ^= h1 >>> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >>> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >>> 16;

    return h1;
  }
  
  /**
   * Implementation of CrapWow Hash, ported from 64-bit version.
   */
  
  public final static int CWOW_32_M = 0x57559429;
  public final static int CWOW_32_N = 0x5052acdb;

  public final static long CWOW_64_M = 0x95b47aa3355ba1a1L;
  public final static long CWOW_64_M_LO = CWOW_64_M & 0x00000000FFFFFFFFL;
  public final static long CWOW_64_M_HI = CWOW_64_M >>> 32;
  
  public final static long CWOW_64_N = 0x8a970be7488fda55L;
  public final static long CWOW_64_N_LO = CWOW_64_N & 0x00000000FFFFFFFFL;
  public final static long CWOW_64_N_HI = CWOW_64_N >>> 32;
  
  public static final long LONG_LO_MASK = 0x00000000FFFFFFFFL;
  
  public static long computeCWowLongHash(byte[] data, long seed) {
      final int length = data.length;
      /* cwfold( a, b, lo, hi ): */
      /* p = (u64)(a) * (u128)(b); lo ^=(u64)p; hi ^= (u64)(p >> 64) */
      /* cwmixa( in ): cwfold( in, m, k, h ) */
      /* cwmixb( in ): cwfold( in, n, h, k ) */

      long hVal = seed;
      long k = length + seed + CWOW_64_N;

      int pos = 0;
      int len = length;

      long aL, aH, bL, bH;
      long r1, r2, r3, rML;
      long pL;
      long pH;

      while (len >= 16) {
          /* cwmixb(X) = cwfold( X, N, hVal, k ) */
          aL = gatherIntLE(data, pos) & LONG_LO_MASK; pos += 4;
          aH = gatherIntLE(data, pos) & LONG_LO_MASK; pos += 4;
          bL = CWOW_64_N_LO; bH = CWOW_64_N_HI;
          r1 = aL * bL; r2 = aH * bL; r3 = aL * bH;
          rML = (r1 >>> 32) + (r2 & LONG_LO_MASK) + (r3 & LONG_LO_MASK);
          pL = (r1 & LONG_LO_MASK) + ((rML & LONG_LO_MASK) << 32);
          pH = (aH * bH) + (rML >>> 32);
          hVal ^= pL; k ^= pH;

          /* cwmixa(Y) = cwfold( Y, M, k, hVal ) */
          aL = gatherIntLE(data, pos) & LONG_LO_MASK; pos += 4;
          aH = gatherIntLE(data, pos) & LONG_LO_MASK; pos += 4;
          bL = CWOW_64_M_LO; bH = CWOW_64_M_HI;
          r1 = aL * bL; r2 = aH * bL; r3 = aL * bH;
          rML = (r1 >>> 32) + (r2 & LONG_LO_MASK) + (r3 & LONG_LO_MASK);
          pL = (r1 & LONG_LO_MASK) + ((rML & LONG_LO_MASK) << 32);
          pH = (aH * bH) + (rML >>> 32);
          k ^= pL; hVal ^= pH;

          len -= 16;
      }

      if (len >= 8) {
          /* cwmixb(X) = cwfold( X, N, hVal, k ) */
          aL = gatherIntLE(data, pos) & LONG_LO_MASK; pos += 4;
          aH = gatherIntLE(data, pos) & LONG_LO_MASK; pos += 4;
          bL = CWOW_64_N_LO; bH = CWOW_64_N_HI;
          r1 = aL * bL; r2 = aH * bL; r3 = aL * bH;
          rML = (r1 >>> 32) + (r2 & LONG_LO_MASK) + (r3 & LONG_LO_MASK);
          pL = (r1 & LONG_LO_MASK) + ((rML & LONG_LO_MASK) << 32);
          pH = (aH * bH) + (rML >>> 32);
          hVal ^= pL; k ^= pH;

          len -= 8;
      }

      if (len > 0) {
          aL = gatherPartialLongLE(data, pos, len);
          aH = aL >> 32;
          aL = aL & LONG_LO_MASK;
          
          /* cwmixa(Y) = cwfold( Y, M, k, hVal ) */
          bL = CWOW_64_M_LO;
          bH = CWOW_64_M_HI;
          r1 = aL * bL; r2 = aH * bL; r3 = aL * bH;
          rML = (r1 >>> 32) + (r2 & LONG_LO_MASK) + (r3 & LONG_LO_MASK);
          pL = (r1 & LONG_LO_MASK) + ((rML & LONG_LO_MASK) << 32);
          pH = (aH * bH) + (rML >>> 32);
          k ^= pL; hVal ^= pH;
      }

      /* cwmixb(X) = cwfold( X, N, hVal, k ) */
      aL = (hVal ^ (k + CWOW_64_N));
      aH = aL >> 32;
      aL = aL & LONG_LO_MASK;
      
      bL = CWOW_64_N_LO;
      bH = CWOW_64_N_HI;
      r1 = aL * bL; r2 = aH * bL; r3 = aL * bH;
      rML = (r1 >>> 32) + (r2 & LONG_LO_MASK) + (r3 & LONG_LO_MASK);
      pL = (r1 & LONG_LO_MASK) + ((rML & LONG_LO_MASK) << 32);
      pH = (aH * bH) + (rML >>> 32);
      hVal ^= pL; k ^= pH;

      hVal ^= k;

      return hVal;
  }
  
  public static final long gatherLongLE(byte[] data, int index) 
  {
      int i1 = gatherIntLE(data, index);
      long l2 = gatherIntLE(data, index + 4);

      return uintToLong(i1) | (l2 << 32);
  }
  
  public static final int gatherIntLE(byte[] data, int index) 
  {
      int i = data[index] & 0xFF;

      i |= (data[++index] & 0xFF) << 8;
      i |= (data[++index] & 0xFF) << 16;
      i |= (data[++index] << 24);

      return i;
  }
  
  public static final long gatherPartialLongLE(byte[] data, int index, int available) 
  {
      if (available >= 4) {
          int i = gatherIntLE(data, index);
          long l = uintToLong(i);

          available -= 4;

          if (available == 0) {
              return l;
          }

          int i2 = gatherPartialIntLE(data, index + 4, available);

          l <<= (available << 3);
          l |= (long) i2;

          return l;
      }

      return (long) gatherPartialIntLE(data, index, available);
  }
  
  public static final int gatherPartialIntLE(byte[] data, int index, int available) 
  {
      int i = data[index] & 0xFF;

      if (available > 1) {
          i |= (data[++index] & 0xFF) << 8;
          if (available > 2) {
              i |= (data[++index] & 0xFF) << 16;
          }
      }

      return i;
  }
  
  public static final long uintToLong(int i) 
  {
      long l = (long) i;
      return (l << 32) >>> 32;
  }
  
}