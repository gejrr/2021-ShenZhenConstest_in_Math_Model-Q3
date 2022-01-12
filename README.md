# 2021深圳杯D题-3
<div>
  <p><a href='http://www.m2ct.org/'>"深圳杯"官网</a></p>
  <p><a href='https://blog.csdn.net/happycodee/article/details/119752470'>2021-D题链接</a></p>
</div>
<h3>题目摘录</h3>
<p>一个古老的羊-犬博弈问题：羊在半径为R的圆形圈内具有定常速率v和满足以下限制的任意转弯能力：逃逸路径上每一点与圆心的距离随时间单调不减。羊逃出圆形圈则胜。犬沿着圆周以定常速率V围堵以防止羊逃逸，任何时刻具有选择圆周的两个方向之一能力。</p>
<p>3. 假设羊理解自己的能力、限制和躲避犬围堵而逃逸目标，但不具备基于运动学的最优化决策知识，假设2中羊可以逃逸的条件被满足，给出一种机器学习方法，使得羊通过学习训练后实现逃逸；</p>
<h3>程序描述</h3>
<div>
  <p>Environment:<br>
  Python v3.7; Tensorflow v2.1; Keras v2.6; numpy v1.21; pygame v2.0.1</p>
<p>gamePacked.py: Consist of the simple game running at either mode: mannual or AI mode. <br>
  Training model: CNN, a Customized Alex-like Net<br>
  Activation function: relu<br>
  Optimizer: Adam <br>
  Loss function: alternative</p>
<p>Controls: <br>
# 用方向键控制羊的移动，↑键沿直径前进，←键逆时针移动，→键顺时针移动。<br>
# 按空格键在人工模式与AI模式间切换。<br>
# 按ESC退出。</p>
<h3>补充说明</h3>
  <p>此题系平时练习，完善度不高，深度学习的效果也一般。在大量训练后，“羊”只学会了“最快速地去世”以提高分数（但仍是负的）；即使读取人工操作的数据加以训练，也会因为一次失败的模仿逐渐走偏，回到原路。期待指点。</p>
  <p>上传的另一个目的，是熟悉git的基本操作。</p>
