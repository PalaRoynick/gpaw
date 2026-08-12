def approx_ipr(calc):
    for kpt for calc.wfs.kpt_u:
        for n in range(10):
            prod = 0.0
            for a in range(10):
                D_ii = np.outer(kpt.P_ani[a][n], kpt.P_ani[a][n])
                D_p = pack(D_ii) # Pack or pack2
                D_pp = np.outer(D_p, D_p)
                prod += (I4_pp * D_pp).sum()
            data[n] = prof

