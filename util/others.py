import numpy as np
from torchvision import transforms
import cv2, torch
from PIL import Image
from pynput import keyboard
from ikpy.chain import Chain
from datetime import datetime
import os

# ------------------------------------------------------------------
# random seed 
def setup_seed(seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------------


def save_txt(opt, losses_list, same_list):
    txt_dir = f"./results/{opt.name}/test_result.txt"
    with open(txt_dir, "a") as txt_file:
        txt_file.write("--------------------------------------\n")
        txt_file.write(f"is_added_DQN: \t{opt.is_added_DQN}\n")
        txt_file.write(f"loss mean:    \t{np.mean(losses_list)}\n")
        txt_file.write(f"loss std:     \t{np.std(losses_list)}\n")
        txt_file.write(f"loss lens:    \t{len(losses_list)}\n")
        txt_file.write(f"loss max:     \t{np.max(losses_list)}\n")
        txt_file.write(f"loss min:     \t{np.min(losses_list)}\n")
        txt_file.write(f"success rate: \t{np.mean(same_list)}\n")
        txt_file.write(f"loss lens: \t\t{len(losses_list)}\n")
        txt_file.write("--------------------------------------\n")

# ----------------------------------------------------------------------------
def split_image(image):
    # image = cv2.imread(image_path, 0)
    a = image[:, :image.shape[0]]
    b = image[:, image.shape[0]:]
    return a, b

def add_black_border_to_square_PIL(image):
    """
    將輸入圖像補上黑邊，使其變成正方形
    :param image: PIL.Image 圖像對象
    :return: 補黑邊後的正方形圖像
    """
    # 獲取原始照片的尺寸
    width, height = image.size
    
    # 計算新照片的邊長（取原始照片的較長邊）
    new_size = max(width, height)
    
    # 創建一個黑色的背景圖像
    new_image = Image.new("RGB", (new_size, new_size), (0, 0, 0))

    # 計算原始照片在新照片中的起始位置
    x_offset = (new_size - width) // 2
    y_offset = (new_size - height) // 2

    # 將原始照片粘貼到黑色背景圖像的中央
    new_image.paste(image, (x_offset, y_offset))

    return new_image

def read_from_PIL(image_path):
    AB = Image.open(image_path)
    # # split AB image into A and B
    w, h = AB.size
    w2 = int(w / 2)
    A = AB.crop((0, 0, w2, h))
    B = AB.crop((w2, 0, w, h))
    return A, B

def get_tensor(image, size=256):
    # 定義變換管道
    transform = transforms.Compose([
        transforms.Lambda(add_black_border_to_square_PIL),
        # transforms.Grayscale(1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
    ])
    return transform(image)

def reverse_transform(transformed_image, original_size=None):
    """
    Reverses the transformations applied to the image and displays it.

    Parameters:
    - transformed_image (torch.Tensor): The transformed image tensor.
    - original_size (tuple): The original size of the image (height, width). Default is None.

    Returns:
    - None
    """
    # Step 1: Remove the extra dimension
    image_tensor = transformed_image.squeeze(0)

    # Step 2: Reverse normalization
    inverse_normalize = transforms.Normalize(
        mean=[-0.5 / 0.5], std=[1 / 0.5]
    )
    image_tensor = inverse_normalize(image_tensor)

    # Step 3: Convert the tensor to a NumPy array
    image_np = image_tensor.squeeze().numpy() * 255  # Scale back to [0, 255]
    image_np = image_np.astype(np.uint8)
    
    return image_np


def cv2_to_pil(cv2_image):
    """
    將 OpenCV 圖像轉換為 PIL 圖像
    :param cv2_image: 使用 OpenCV (cv2) 讀取或處理的 numpy 圖像
    :return: PIL.Image 圖像對象
    """
    # 檢查圖像是否為灰階
    if len(cv2_image.shape) == 2:
        # 灰階圖像直接轉換
        pil_image = Image.fromarray(cv2_image)
    else:
        # 彩色圖像需要轉換顏色通道順序 (從 BGR 轉為 RGB)
        cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_image_rgb)
    
    return pil_image
    
# ----------------------------------------------------------------------------

def is_outside_area(x, point_min, point_max):
    try:
        return not((point_min[0] <= x[0] <= point_max[0]) \
            and (point_min[1] <= x[1] <= point_max[1]))
    except:
        return not(point_min <= x <= point_max)
    finally:
        pass

def is_insafe_area_on_table_xy(position_xy):
    arm_tolerance = 0.03
    arm_area_min = [-0.095-arm_tolerance, -0.14-arm_tolerance]
    arm_area_max = [0.095+arm_tolerance, 0.14+arm_tolerance]
    
    stuff_torolerance = 0.05
    back_stuff_min = [-0.325, -0.29-stuff_torolerance]
    back_stuff_max = [-0.145+stuff_torolerance, 0.82]
    
    right_hand_min = [-0.145, 0.42-stuff_torolerance]
    right_hand_max = [0.115+stuff_torolerance, 0.82]
    
    working_area_min = [-0.225, -0.68]
    working_area_max = [0.445, 0.72]
    
    return \
        is_outside_area(position_xy, back_stuff_min, back_stuff_max) \
        and is_outside_area(position_xy, right_hand_min, right_hand_max) \
        # is_outside_area(position_xy, arm_area_min, arm_area_max) \
        # and not(is_outside_area(position_xy, working_area_min, working_area_max))

def check_next_position_is_safe(position_xyz):
    if not(is_insafe_area_on_table_xy(position_xyz[:2])) \
        and position_xyz[2] <= 0.2:
        print("will collide with stuff")
        return False
    
    
    working_area_min = [-0.225, -0.68]
    working_area_max = [0.445, 0.72]
    if is_outside_area(position_xyz[:2], working_area_min, working_area_max):
        print("outside working area")
        return False
    
    if is_outside_area(position_xyz[2], 0.08, 0.9):
        print("outside working height")
        return False
    
    return True

def calculate_distance(a_position, b_position):
    assert len(a_position) == len(b_position), "Error: length not the same, cant calculate"
    
    tem = 0
    for i in range(len(a_position)):
        tem += (a_position[i] - b_position[i]) ** 2
    
    distance = tem ** 0.5

    return distance 


def check_is_success(position_xyz, target_xyz, lift_height=0.12, success_distance=0.05, success_type="distance"):
    target_xyz_ = target_xyz.copy()
    target_xyz_[2] = target_xyz_[2] + lift_height
    
    if success_type == "distance":
        dis = calculate_distance(position_xyz, target_xyz_)    
        return dis <= success_distance
    
    elif success_type == "block":
        target_min = np.asarray(target_xyz_) - success_distance
        target_max = np.asarray(target_xyz_) + success_distance
        return is_in_range_xyz(position_xyz, target_min, target_max)
    
    else:
        return False

def is_in_range_xyz(xyz, xyz_min, xyz_max):
    return xyz_min[0] <= xyz[0] <= xyz_max[0] \
        and xyz_min[1] <= xyz[1] <= xyz_max[1] \
        and xyz_min[2] <= xyz[2] <= xyz_max[2]
        
    
    
def get_frame(my_cam, event, model_name):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('video/output.avi', fourcc, 30.0, (1280, 720))
    current_time = datetime.now().strftime("%b%d_H%H_M%M_S%S")
    print("current_time", current_time)
    os.makedirs(rf'D:\research_video\video\{model_name}', exist_ok=True)
    out = cv2.VideoWriter(rf'D:\research_video\video\{model_name}\{current_time}_output.avi', fourcc, 30.0, (1280, 720))
    
    while not event.is_set():
        frame = my_cam.get_frame()
        # 寫入視頻文件
        out.write(frame)
        
        cv2.imshow('RealSense', frame)
        # 檢查是否按下 'q' 鍵來退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("get_frame received stop signal.")    
    out.release()  
    
def detect_success(event, pause_event, target_position, success_distance, Mem):
    while not event.is_set():
        arm_position = Mem.get_arm_position()
        # print(arm_position)
        
        dis = calculate_distance(arm_position, target_position)
        
        if dis <= success_distance:
            print("Success from thread!")
            event.set()
            break 
        while pause_event.is_set():
            pass               
        
    print("Worker received stop signal.")
    


def on_press(key):
    try:
        key_char = key.char.lower()
        print(f"按下的鍵: {key_char}")
        if key_char == 'y':
            print("'y' 被按下，程序結束。")
            return False  # 停止監聽
    except AttributeError:
        print(f"按下的特殊鍵: {key}")

def detect_y():
    # 開始監聽
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    print("程序已退出。")  
    
def read_urdf(urdf_path):
    return Chain.from_urdf_file(urdf_path)

def check_wrist_flip(chain, target_position, target_orientation=[0, 0, -1], orientation_mode="Z"):
    ik = chain.inverse_kinematics(target_position, 
                                  target_orientation, 
                                  orientation_mode)
    
    return ik[-2] <= 0