
原有两个py文件，Figure_2.py和Function_Figure_2.py
前者调用后者

其中为gmt绘图，版本gmt4 (非常旧的版本了，现在已经到gmt6了)

问题：
1. 系统 代码未Ubuntu系统下的命令，在windows下执行没有权限；
2. 版本 gmt4到gmt6有显著不同，gmt4繁琐复杂会生成多个中间文件，并被称为经典模式，gmt6为现代模式，许多代码需要修改；
    2.1 下载gmt4运行 (不建议，gmt4繁琐复杂，虽然有现成代码，但是已经逐渐被淘汰，顺应趋势应用gmt6现代模式才是)
    2.2 将gmt4代码改成gmt6 需要学习gmt代码，理解每段gmt4代码做什么，再用gmt6写出来执行
3. Python调用执行问题 Python调用执行在windows下有问题，gmt basemap命令找不到，可能是环境变量等问题
    3.1 可以放弃Python调用，直接使用写入.sh文件，使用git bash调用(git bash可以执行.sh)
    3.2 在Ubuntu系统下使用

gmt_Figure2.sh 为gmt6现代模式的代码