import numpy as np
import math as m



collateral_orig = 117_839_000       # original collateral
collateral_remain = 84_096_204      # remaining collateral
collateral_remain_copy = 84_096_000
tranche_A_notional = 63_255_947     # remaining tranche A notional
tranche_A_notional_copy = 63_255_947
z_notional = 20840256

coupon = 0.055                       # annual coupon
wam_remaining = 360-33                  # remaining weighted average maturity (months)
psa = 315                             # PSA percentage
ytm = 0.04489                         # annual YTM for discounting
wac = 0.06474

pmt_month_coupon = wac/12

months = np.arange(1, 361)
month_coupon = coupon/12

cpr_100 = np.minimum(0.06 * months / 30, 0.06)
cpr_psa = cpr_100 * psa / 100
smm = 1 - (1 - cpr_psa)**(1/12)

adjusted_smm = smm[35:]



PMT = ((pmt_month_coupon * (1 + pmt_month_coupon) ** 358) / (((1 + pmt_month_coupon) ** 358) - 1)) * collateral_orig

cf = []
i = 0


while collateral_remain > (collateral_remain_copy - tranche_A_notional):


    total_interest = collateral_remain * pmt_month_coupon
    principal_paid = PMT - total_interest

    z_interest = z_notional * month_coupon

    a_interest = max(0,(collateral_remain*month_coupon) - z_interest)
    # this max function, helps make sure a_interest does not go below fucking 0 in last feew months 
    # in fact it pushes price from 101.7 ---> 101.75, which make the model 100% accurate, hell yeah


    updated_pool_balance = collateral_remain - principal_paid

    pre_pay_amount = updated_pool_balance * adjusted_smm[i]

    updated_pool_balance -= z_interest

    updated_pool_balance -= pre_pay_amount

    #  UPDATES
    z_notional += z_interest
    collateral_remain = updated_pool_balance
    tranche_A_notional_copy -= (pre_pay_amount + principal_paid + z_interest)
    # CASH FLOWS

    investor_a_cf = z_interest + a_interest + principal_paid + pre_pay_amount


    cf.append(investor_a_cf)

    i += 1

cf[-1] += (collateral_remain_copy - tranche_A_notional) - collateral_remain

print(collateral_remain)
print(z_notional)
print(i)

print(cf)

dcf = 0

j = 1
for i in cf:
    dcf += i/((1+ytm)**(1/12 * (j-1+24/30)))
    j += 1

print(dcf)
print('price')
print(dcf/tranche_A_notional * 100)
# bid/ask target price is 101.75

