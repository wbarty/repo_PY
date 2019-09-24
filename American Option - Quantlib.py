import QuantLib as ql

# global data
t = ql.Date(DD, ql.MM, YYYY)
ql.Settings.instance().evaluationDate = t
T = ql.Date(DD, ql.MM, YYYY)
r = ql.FlatForward(T, 0.014, ql.Actual365Fixed())

# option parameters
exercise = ql.AmericanExercise(T, ql.Date(DD, ql.MM, YYYY))
payoff = ql.PlainVanillaPayoff(ql.Option.Put, 40.0)

# market data
S = ql.SimpleQuote($S)
v = ql.BlackConstantVol(t, ql.TARGET(), 0.20, ql.Actual365Fixed())
q = ql.FlatForward(T, 0.00, ql.Actual365Fixed())

# Results table
header = "%19s" % "method" + " |" + " |".join(["%17s" % tag for tag in ["value", "estimated error", "actual error"]])
print("")
print(header)
print("-" * len(header))
refValue = None


def report(method, x, dx=None):
    e = "%.4f" % abs(x - refValue)
    x = "%.5f" % x
    if dx:
        dx = "%.4f" % dx
    else:
        dx = "n/a"
    print("%19s" % method + " |" + " |".join(["%17s" % y for y in [x, dx, e]]))


# Option Process
process = ql.BlackScholesMertonProcess(
    ql.QuoteHandle(S),
    ql.YieldTermStructureHandle(q),
    ql.YieldTermStructureHandle(r),
    ql.BlackVolTermStructureHandle(v),
)

option = ql.VanillaOption(payoff, exercise)

refValue = $reference value$
report("reference value", refValue)

# method - analytic
option.setPricingEngine(ql.BaroneAdesiWhaleyEngine(process))
report("Barone-Adesi-Whaley", option.NPV())

option.setPricingEngine(ql.BjerksundStenslandEngine(process))
report("Bjerksund-Stensland", option.NPV())

# method - finite differences
timeSteps = 801
gridPoints = 800

option.setPricingEngine(ql.FDAmericanEngine(process, timeSteps, gridPoints))
report("finite differences", option.NPV())

# method - binomial
timeSteps = 801

option.setPricingEngine(ql.BinomialVanillaEngine(process, "JR", timeSteps))
report("binomial (JR)", option.NPV())

option.setPricingEngine(ql.BinomialVanillaEngine(process, "CRR", timeSteps))
report("binomial (CRR)", option.NPV())

option.setPricingEngine(ql.BinomialVanillaEngine(process, "EQP", timeSteps))
report("binomial (EQP)", option.NPV())

option.setPricingEngine(ql.BinomialVanillaEngine(process, "Trigeorgis", timeSteps))
report("bin. (Trigeorgis)", option.NPV())

option.setPricingEngine(ql.BinomialVanillaEngine(process, "Tian", timeSteps))
report("binomial (Tian)", option.NPV())

option.setPricingEngine(ql.BinomialVanillaEngine(process, "LR", timeSteps))
report("binomial (LR)", option.NPV())

option.setPricingEngine(ql.BinomialVanillaEngine(process, "Joshi4", timeSteps))
report("binomial (Joshi)", option.NPV())
