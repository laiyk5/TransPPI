import torch
from torch.optim import lr_scheduler as lrs
from torch.nn import Linear
from torch.optim import Adam

model = Linear(128, 1)
opt = Adam(model.parameters(), lr=1e-3)
# scheduler = lrs.ConstantLR(opt, factor=1.0, verbose=True)
scheduler = lrs.StepLR(opt, step_size=3, gamma=0.1, verbose=True)

for i in range(20):
    scheduler.step()
    print(scheduler.get_lr())