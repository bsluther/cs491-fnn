
### Forward/Backward State

<code>
Note: this is rather wide and if it wordwraps it won't make any sense.

K: total # of layers - 1 (because we're zero-indexed)
k: the current layer
a[k]: preactivation vector of the kth layer
h[k]: postactivation vector of the kth layer
g_a[k]: loss-to-node gradient for activation values of the kth layer, eg dL/da_k
g_h[k]: loss-to-node gradient for postactivation values of the kth layer, eg dL/dh_k

    W[0] @ x         phi(a[0])        W[1] @ h[0]       phi(a[2])      W[2] @ h[1]     W[n] @ h[n-1]     phi(a[n])             L(h[n]) 

 x   ------>   a[0]   ------>   h[0]   ------>   a[1]   ------>   h[1]   ------>   ...   ------>   a[n]   ------>   h[n] = o   ------>   L
     
              g_a[0]           g_h[0]           g_a[1]           g_h[1]                           g_a[n]           g_h[n]               dL/do
</code>  