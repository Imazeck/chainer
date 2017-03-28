from chainer import FunctionSet,Variable
import chainer.functions as F
import numpy
import numpy as np
import pdb
from chainer import cuda


# VAEの学習epoch
num_epochs = 100
# VAEモデルの学習のミニバッチサイズ
batchsize = 100

# MNISTモデルのバッチサイズ
batchsize_ = 100
# MNISTモデルの学習の繰り返し回数
n_epoch   = 20
# MNISTモデルの中間層の数
n_units   = 1000
# 表示するdigit数(n*10)
n_show = 20

class VAE_gaussian(FunctionSet):

    # You must add attr named with 'enc_l1' and 'dec_l1'
    # for specification of the input and latent dimensions, in __init__ of FunctionSet.

    def encode(self,x):
        raise NotImplementedError()

    def decode(self,z):
        raise NotImplementedError()

    def free_energy(self,x):
        #return -(free energy)
        enc_mu, enc_log_sigma_2 = self.encode(x)
        kl = F.gaussian_kl_divergence(enc_mu,enc_log_sigma_2)
        z = F.gaussian(enc_mu,enc_log_sigma_2)
        dec_mu, dec_log_sigma_2 = self.decode(z)
        nll = F.gaussian_nll(x,dec_mu,dec_log_sigma_2)
        return nll+kl

    def generate(self,N,sampling_x=False):
        z_dim = self['dec_l1'].W.shape[1]
        if(isinstance(self['dec_l1'].W,np.ndarray)):
            zero_mat = Variable(np.zeros((N,z_dim),'float32'))
            z = F.gaussian(zero_mat,zero_mat)
        else:
            raise NotImplementedError()
        dec_mu, dec_log_sigma_2 = self.decode(z)
        if(sampling_x):
            x = F.gaussian(dec_mu,dec_log_sigma_2)
        else:
            x = dec_mu
        return x

    def reconstruct(self,x):
        enc_mu, enc_log_sigma_2 = self.encode(x)
        dec_mu, dec_log_sigma_2 = self.decode(enc_mu)
        return dec_mu

class VAE_bernoulli(VAE_gaussian):

    def free_energy(self,x):
        #return -(free energy)
        enc_mu, enc_log_sigma_2 = self.encode(x)
        kl = F.gaussian_kl_divergence(enc_mu,enc_log_sigma_2)
        z = F.gaussian(enc_mu,enc_log_sigma_2)
        dec_mu = self.decode(z)
        nll = F.bernoulli_nll(x,dec_mu)
        return nll+kl

    def generate(self,N,sampling_x=False):
        z_dim = self['dec_l1'].W.shape[1]
        if(isinstance(self['dec_l1'].W.data,np.ndarray)):
            zero_mat = Variable(np.zeros((N,z_dim),'float32'))
            z = F.gaussian(zero_mat,zero_mat)
        else:
            raise NotImplementedError()

        dec_mu = F.sigmoid(self.decode(z))
        if(sampling_x):
            raise NotImplementedError()
        else:
            x = dec_mu
        return x


    def reconstruct(self,x):
        enc_mu, enc_log_sigma_2 = self.encode(x)
        dec_mu = F.sigmoid(self.decode(enc_mu))
        return dec_mu


from chainer import FunctionSet,Variable,serializers
import chainer.functions as F
import numpy


class VAE_MNIST(VAE_gaussian):
    def __init__(self):
        super(VAE_MNIST,self).__init__(
            enc_l1 = F.Linear(784,500),
            enc_l_mu = F.Linear(500,30),
            enc_l_log_sig_2 = F.Linear(500,30),
            dec_l1 = F.Linear(30,500),
            dec_l_mu = F.Linear(500,784),
            dec_l_log_sig_2 = F.Linear(500,784))

    def encode(self,x):
        h = x
        h = self.enc_l1(h)
        h = F.relu(h)
        return self.enc_l_mu(h),self.enc_l_log_sig_2(h)

    def decode(self,z):
        h = z
        h = self.dec_l1(h)
        h = F.relu(h)
        return self.dec_l_mu(h),self.dec_l_log_sig_2(h)

class VAE_MNIST_b(VAE_bernoulli):
    def __init__(self):
        super(VAE_MNIST_b,self).__init__(
            enc_l1 = F.Linear(784,500),
            enc_l2 = F.Linear(500,500),
            enc_l_mu = F.Linear(500,30),
            enc_l_log_sig_2 = F.Linear(500,30),
            dec_l1 = F.Linear(30,500),
            dec_l2 = F.Linear(500,500),
            dec_l_mu = F.Linear(500,784))

    def encode(self,x):
        h = x
        h = self.enc_l1(h)
        h = F.relu(h)
        h = self.enc_l2(h)
        h = F.relu(h)
        return self.enc_l_mu(h),self.enc_l_log_sig_2(h)
    
    def decode(self,z):
        h = z
        h = self.dec_l1(h)
        h = F.relu(h)
        h = self.dec_l2(h)
        return self.dec_l_mu(h)
    

from chainer import FunctionSet,Variable

import chainer.functions as F
import chainer.optimizers as optimizers
import numpy
import matplotlib.pyplot as plt

vae = VAE_MNIST_b()
optimizer = optimizers.Adam(alpha=0.001)
optimizer.setup(vae)

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
x_all = mnist.data.astype(np.float32)/255
#y_all = mnist.target.astype(np.int32)
x_train, x_test = np.split(x_all, [60000])
#y_train, y_test = np.split(y_all, [60000])



train_free_energies = np.zeros(num_epochs)
test_free_energies = np.zeros(num_epochs)


for epoch in range(num_epochs):
    indexes = np.random.permutation(60000)
    n_batch = indexes.shape[0]/batchsize
    sum_free_energy = 0

    for i in range(0, 60000, batchsize):
        x_batch = Variable(np.asarray(x_train[indexes[i : i + batchsize]]))
        free_energy = vae.free_energy(x_batch)
        
        #print(free_energy.data)
        sum_free_energy += free_energy.data
        optimizer.zero_grads()
        free_energy.backward()
        optimizer.update()
    
    indexes = np.random.permutation(10000)   
    sum_test_free_energy = 0
    for i in range(0, 10000, batchsize):
        x_batch = Variable(np.asarray(x_test[indexes[i : i + batchsize]]))
        free_energy = vae.free_energy(x_batch)
        sum_test_free_energy += free_energy.data

    train_free_energies[epoch] = sum_free_energy/60000
    test_free_energies[epoch] = sum_test_free_energy/10000
         
    print( '[epoch ' +  str(epoch) +']')
    print ('train free energy:' + str(train_free_energies[epoch]))
    print ('test free energy:' + str(test_free_energies[epoch]))

fontsize = 14
plt.plot(np.arange(0,num_epochs),cuda.to_cpu(train_free_energies),label='train free energy')
plt.plot(np.arange(0,num_epochs),cuda.to_cpu(test_free_energies),label='test free energy')
plt.legend(fontsize=fontsize)
plt.xlabel('epoch',fontsize=fontsize)
plt.ylabel('$F(X,\\theta,\phi)$',fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
#plt.show()


# In[2]:


'''serializers.load_npz('my.model', vae)
serializers.load_npz('my.state', optimizer)
'''


# In[3]:

'''
serializers.save_npz('my.model', vae)
serializers.save_npz('my.state', optimizer)
'''


# In[4]:

x_samples = vae.generate(10000,False)
'''
plt.figure()
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(cuda.to_cpu(x_samples.data[i].reshape(28,28)),'binary')
plt.show()


# In[5]:

indexes = np.random.permutation(10000)   
x_batch = Variable(np.asarray(x_test[indexes[:10]]))
x_recon = vae.reconstruct(x_batch)
plt.figure()
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(cuda.to_cpu(x_batch.data[i].reshape(28,28)),'binary')
for i in range(5):
    plt.subplot(2,5,i+6)
    plt.imshow(cuda.to_cpu(x_recon.data[i].reshape(28,28)),'binary')
plt.show()
'''

# In[6]:



print ('fetching MNIST dataset')
mnist = fetch_mldata('MNIST original')
mnist.data   = mnist.data.astype(np.float32)
mnist.data  /= 255  
mnist.target = mnist.target.astype(np.int32)

# 学習用データを N個、検証用データを残りの個数と設定
N = 60000
x_train, x_test = np.split(mnist.data,   [N])
y_train, y_test = np.split(mnist.target, [N])
N_test = y_test.size

model = FunctionSet(l1=F.Linear(784, n_units),
                    l2=F.Linear(n_units, n_units),
                    l3=F.Linear(n_units, 10))

def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y  = model.l3(h2)
    
    y_prob = F.softmax(y)
    entropy = - F.sum((F.log2(y_prob) * y_prob))
    if train:
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
    if not train:
        return entropy


optimizer2 = optimizers.Adam()
optimizer2.setup(model)

train_loss = []
train_acc  = []
test_loss = []
test_acc  = []

# Learning loop
for epoch in range(1, n_epoch+1):
    print ('epoch', epoch)

    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N, batchsize_):
        x_batch = np.asarray(x_train[perm[i:i+batchsize]])
        y_batch = np.asarray(y_train[perm[i:i+batchsize]])

        optimizer2.zero_grads()
        loss, acc = forward(x_batch, y_batch)
        loss.backward()
        optimizer2.update()

        train_loss.append(loss.data)
        train_acc.append(acc.data)
        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    print ('train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N))


# In[7]:

'''
serializers.load_npz('my.model_mnist', model)
serializers.load_npz('my.state_mnist', optimizer2)
'''


# In[8]:

'''
serializers.save_npz('my.model_mnist', model)
serializers.save_npz('my.state_mnist', optimizer2)
'''


# In[9]:

print('calculating entropy')
entropy_list = np.zeros(10000)
x_test_samples = x_samples.data
batchsize = 1
for i in range(0, 10000, batchsize):
    x_batch = np.asarray(x_test_samples[i:i+batchsize])
    y_batch = np.asarray(y_test[i:i+batchsize])
    entropy = forward(x_batch, y_batch, train=False)
    entropy_list[i] = entropy.data
entropy_list = np.nan_to_num(cuda.to_cpu(entropy_list))   
entropy_list2 = np.sort(cuda.to_cpu(entropy_list))[::-1]
entropy_list3 = np.sort(cuda.to_cpu(entropy_list))
max_entropy = entropy_list2[0:n_show]
min_entropy = entropy_list3[0:n_show]
index_list = np.zeros(n_show,dtype=np.int64)
index_list2 = np.zeros(n_show,dtype=np.int64)


for j in range(n_show):
    index = np.where(cuda.to_cpu(entropy_list) == max_entropy[j])
    #pdb.set_trace()
    index_list[j] = index[0][0]
    
    index = np.where(cuda.to_cpu(entropy_list) == min_entropy[j])
    index_list2[j] = index[0][0]

    

plt.figure(figsize=(15,15))
for i in range(n_show):
    ind = index_list[i]
    plt.subplot(n_show/5,5,i+1)
    plt.imshow(cuda.to_cpu(x_test_samples[ind].reshape(28,28)),'binary')
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")


# In[10]:

plt.figure(figsize=(15,15))
for i in range(n_show):
    ind = index_list2[i]
    plt.subplot(n_show/5,5,i+1)
    plt.imshow(cuda.to_cpu(x_test_samples[ind].reshape(28,28)),'binary')
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")
    
plt.show()

