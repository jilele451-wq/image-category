import cv2
from PIL import Image
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.nn as nn

tsfrm = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 0.水果 1.蔬菜 2.服装 3.零食
classes = ('水果','蔬菜','服装','零食')

# 判断GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 模型推理
def modelpre(grey_img):
    model_eval = models.resnet18(pretrained=False)
    num_ftrs = model_eval.fc.in_features
    model_eval.fc = nn.Linear(num_ftrs, 4)
    model_eval.load_state_dict(torch.load('model\Model-20230319.pkl', map_location=device))
    # 在推理前，务必调用model.eval()去设置dropout和batch normalization层为评估模式
    model_eval.eval()
    # OpenCV转PIL格式
    image = Image.fromarray(cv2.cvtColor(grey_img, cv2.COLOR_GRAY2RGB))
    # PIL图像数据转换为tensor数据，并归一化
    img = tsfrm(image)
    # 图像增加1维[batch_size,通道,高,宽]
    img = img.unsqueeze(0)
    # 输出推理结果
    output = model_eval(img)
    # prob是4个分类的概率
    prob = F.softmax(output, dim=1)
    print(prob)
    value, predicted = torch.max(output.data, 1)
    label = predicted.numpy()[0]
    print(label)
    pred_class = classes[predicted.item()]
    print(pred_class)

    # 在图像上添加分类结果
    result_img = grey_img.copy()
    cv2.putText(result_img, f"Class: {pred_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 保存结果图像
    cv2.imwrite(f'result/result_{pred_class}.jpg', result_img)

if __name__ == "__main__":
    img_source = cv2.imread('./image/161.jpg')
    # 图像灰度化
    grey_img = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    cv2.imshow("image", grey_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    modelpre(grey_img)



