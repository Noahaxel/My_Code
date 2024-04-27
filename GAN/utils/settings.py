import os
filepath = 'C:\\Users\\ASUS\\Desktop\\ANN_project\\GAN' # 根目录
route = filepath + '\\data\\Stanford_Dogs\\Images'  # 数据目录
result_save_path= filepath + '\\model\\GAN_model'  #训练好的生成网络模型的目录
fakedata_save_path= filepath + '\\data\\Stanford_Dogs\\fake_image'   #生成的fake图片保存目录
route = filepath + '\\data\\Stanford_Dogs\\Images'  # 真实图片目录
Sorts = os.listdir(route)	#分类种类