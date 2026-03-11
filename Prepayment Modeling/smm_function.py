
def smm_function(psi_0, refi_path, turnover_path):

    # You can start with psi_0 with a value of 0.5

    beta = 0.01
    psi = None

    smm_path = []

    active_lambdas = []

    for i in range(len(refi_path)):

        lambda_active_i = refi_path[i] + turnover_path[i]
        lambda_passive_i = refi_path[i] + (beta*turnover_path[i])

        active_lambdas.append(lambda_active_i)

        if psi is None:
            psi = psi_0
        else:
            # psi defined up to this point is psi used in previous iteration
            # new psi is based on previous full_smm, previous psi, and previous active smm
            psi = (psi*(1-active_lambdas[-1])) / (1-smm_path[-1])

        full_smm = psi*lambda_active_i + (1-psi)*lambda_passive_i

        smm_path.append(full_smm)



    return smm_path
