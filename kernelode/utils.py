import torch


def predict_class_vector(gpr, X_test):
    X = gpr.X
    Y = gpr.y
    N = gpr.X.size(0)            
    Kff = gpr.kernel(X).contiguous()
    Kff.view(-1)[:: N + 1] += gpr.jitter + gpr.noise 
    KinvY = torch.solve(Y.T.float(), Kff).solution # might be good to replace with lstsq, or some chosolve 
#     import pdb;pdb.set_trace()
    Ktest = gpr.kernel(X_test, X).contiguous()
    return Ktest.mm(KinvY)

def predict_class(gpr, X_test):
    class_vector = predict_class_vector(gpr, X_test)
    return torch.argmax(class_vector, axis=1)