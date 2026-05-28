step1:
1.target_pos_ratio 更低了;
2.pos_floor_loss 太弱，基本没起到约束作用;
3.candidate cap / anchor cap / prev keep 都没有 active，说明不是正则压住，而是 source gate 没被 positive supervision 推开;
5.state_delta_abs 主要来自 source mixture，而不是 residual delta。residual 路径基本没有参与;
6.relation attention 几乎是强对角自关注,基本退化成逐 token 独立处理。原因很可能是当前 token-level supervision 太弱，relation block 没有被驱动;
7.pos ratio系列指标全部都是：0.05067这看起来不像“四个 AND 条件中哪个最苛刻”的诊断，而更像是经过top-ratio 或最终 pos mask 后的条件通过率。
step2:
1.soft target/focal 提升了 positive 权重的绝对值，
但还没有学出 positive > negative 的选择性吸收；
2.focal 的实际降权效果不明显，或者当前样本还没有形成大量 easy negative；从 loss 数值看，现在 focal 还没有真正改变优化结构。它更像是 soft target 扩大了监督，而不是 focal 起了主要作用。