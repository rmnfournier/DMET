%% analyze the output of train_gans.py


%% Compare the distribution of the GAN output with the dataset
L_gan = importdata("L.csv");
L_data = importdata("../data_L2_N2_sym/data.csv");

m_gan = mean(L_gan);
v_gan = var(L_gan);

m_data = mean(L_data);
v_data = var(L_data);

figure(1)
errorbar(1:21,m_gan,sqrt(v_gan))
hold on
errorbar(1:21,m_data,sqrt(v_data))
xlabel("ith. coding element")
ylabel("mean value +- std dev")
legend("gan","data")

figure(2)
subplot(1,2,1)
heatmap(abs(corr(L_gan)))
title("correlation matrix gan")
subplot(1,2,2)
heatmap(abs(corr(L_data)))
title("correlation matrix data")


%% History loss 
loss_g = importdata("Loss_g.csv");
loss_d = importdata("Loss_d.csv");
figure(3)
lg = length(loss_g);
plot(loss_g)
hold on 
ld=length(loss_d);
plot(loss_d)
plot(1:ld,repmat(log(2),ld,1))
legend("Loss generator","Loss discriminator","50/50 target")
xlabel("Epoch")
ylabel("Cross entropy")