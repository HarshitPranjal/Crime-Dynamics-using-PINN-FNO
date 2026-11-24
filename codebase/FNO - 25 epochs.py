import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.fft
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

plt.rcParams.update({
    "figure.dpi":300,
    "savefig.dpi":600,
    "font.family":"serif",
    "font.size":13,
    "axes.titlesize":15,
    "axes.labelsize":14,
    "savefig.bbox":"tight"
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

inputs = np.load("crime_fno_dataset/inputs.npy")
targets = np.load("crime_fno_dataset/targets.npy")

class CrimeDataset(Dataset):
    def __init__(self,x,y):
        self.x=torch.tensor(x,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

X_train,X_test,Y_train,Y_test = train_test_split(
    inputs,targets,test_size=0.2,random_state=42
)

train_loader=DataLoader(CrimeDataset(X_train,Y_train),batch_size=8,shuffle=True)
test_loader=DataLoader(CrimeDataset(X_test,Y_test),batch_size=8,shuffle=False)

class SpectralConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,modes1,modes2):
        super().__init__()
        self.modes1=modes1
        self.modes2=modes2
        scale=1/(in_channels*out_channels)
        self.weights=nn.Parameter(
            scale*torch.randn(
                in_channels,out_channels,modes1,modes2,
                dtype=torch.cfloat
            )
        )

    def compl_mul2d(self,input,weights):
        return torch.einsum("bixy,ioxy->boxy",input,weights)

    def forward(self,x):
        B=x.shape[0]
        x_ft=torch.fft.rfft2(x)

        out_ft=torch.zeros(
            B,self.weights.shape[1],
            x.size(-2),
            x.size(-1)//2+1,
            dtype=torch.cfloat,
            device=x.device
        )

        out_ft[:,:,:self.modes1,:self.modes2] = self.compl_mul2d(
            x_ft[:,:,:self.modes1,:self.modes2],
            self.weights
        )

        return torch.fft.irfft2(out_ft,s=(x.size(-2),x.size(-1)))

class FNO2D(nn.Module):
    def __init__(self,modes=16,width=64):
        super().__init__()

        self.fc0=nn.Conv2d(10,width,1)

        self.conv0=SpectralConv2d(width,width,modes,modes)
        self.conv1=SpectralConv2d(width,width,modes,modes)
        self.conv2=SpectralConv2d(width,width,modes,modes)
        self.conv3=SpectralConv2d(width,width,modes,modes)

        self.w0=nn.Conv2d(width,width,1)
        self.w1=nn.Conv2d(width,width,1)
        self.w2=nn.Conv2d(width,width,1)
        self.w3=nn.Conv2d(width,width,1)

        self.bn0=nn.BatchNorm2d(width)
        self.bn1=nn.BatchNorm2d(width)
        self.bn2=nn.BatchNorm2d(width)
        self.bn3=nn.BatchNorm2d(width)

        self.head=nn.Conv2d(width,3,1)

    def forward(self,x):
        B,_,H,W=x.shape

        gx=torch.linspace(0,1,W,device=x.device).view(1,1,1,W).repeat(B,1,H,1)
        gy=torch.linspace(0,1,H,device=x.device).view(1,1,H,1).repeat(B,1,1,W)

        x=torch.cat([x,gx,gy],dim=1)
        x=self.fc0(x)

        x=torch.relu(self.bn0(self.conv0(x)+self.w0(x)))
        x=torch.relu(self.bn1(self.conv1(x)+self.w1(x)))
        x=torch.relu(self.bn2(self.conv2(x)+self.w2(x)))
        x=torch.relu(self.bn3(self.conv3(x)+self.w3(x)))

        x=self.head(x)
        return x

model=FNO2D().to(device)

optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-6)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,25,0.5)

def relative_l2_loss(pred,true):
    return torch.norm(pred-true)/torch.norm(true)

epochs=25
best=np.inf
train_hist=[]
val_hist=[]

for epoch in range(epochs):
    model.train()
    tr=0.0

    for x,y in train_loader:
        x,y=x.to(device),y.to(device)

        optimizer.zero_grad()
        pred=model(x)
        loss=relative_l2_loss(pred,y)
        loss.backward()
        optimizer.step()

        tr+=loss.item()

    tr/=len(train_loader)
    train_hist.append(tr)

    model.eval()
    vl=0.0

    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device),y.to(device)
            vl+=relative_l2_loss(model(x),y).item()

    vl/=len(test_loader)
    val_hist.append(vl)

    if vl<best:
        best=vl
        torch.save(model.state_dict(),"best_crime_fno.pt")

    scheduler.step()

    print(f"Epoch {epoch+1:03d} | Train {tr:.6e} | Val {vl:.6e}")

fig,ax=plt.subplots(figsize=(7,5))
ax.semilogy(train_hist,label="Training",lw=2)
ax.semilogy(val_hist,label="Validation",lw=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Relative L2 Loss")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
plt.savefig("fno_loss.pdf")
plt.savefig("fno_loss.png")
plt.show()

model.load_state_dict(torch.load("best_crime_fno.pt",map_location=device))
model.eval()

x,y=next(iter(test_loader))

start=time.time()
with torch.no_grad():
    pred=model(x.to(device)).cpu().numpy()
elapsed=time.time()-start

truth=y.numpy()

rmse=np.sqrt(np.mean((pred-truth)**2))
rel=np.linalg.norm(pred-truth)/np.linalg.norm(truth)

print("RMSE =",rmse)
print("Relative L2 =",rel)
print("Inference Time =",elapsed)

sample=0
# The output of FNO has dimensions (batch, channels, height, width)
# Channels correspond to S, C, P. Criminal population (C) is at index 1.
channel_index_C = 1

vmin=min(truth[sample, channel_index_C, :, :].min(),pred[sample, channel_index_C, :, :].min())
vmax=max(truth[sample, channel_index_C, :, :].max(),pred[sample, channel_index_C, :, :].max())

fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)

im0=axes[0].imshow(truth[sample, channel_index_C, :, :],origin="lower",
                   cmap="viridis",vmin=vmin,vmax=vmax)
axes[0].set_title("FDM Reference (C population)")

im1=axes[1].imshow(pred[sample, channel_index_C, :, :],origin="lower",
                   cmap="viridis",vmin=vmin,vmax=vmax)
axes[1].set_title("FNO Prediction (C population)")

err=np.abs(truth[sample, channel_index_C, :, :]-pred[sample, channel_index_C, :, :])
im2=axes[2].imshow(err,origin="lower",cmap="viridis")
axes[2].set_title("Absolute Error (C population)")

for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

plt.colorbar(im0,ax=axes[0])
plt.colorbar(im1,ax=axes[1])
plt.colorbar(im2,ax=axes[2])

plt.savefig("fdm_vs_fno.pdf")
plt.savefig("fdm_vs_fno.png")
plt.show()
