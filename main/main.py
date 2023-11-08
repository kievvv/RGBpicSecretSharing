from PIL import Image
import numpy as np

#--------------------------------参数设定-------------------------------
n = 5 # 秘密分割份数
r = 3 # 门限数
path = 'test4.png' # 秘密图片路径
testNum = 4 # 用于测试图片的命名
mod = 257 # 模数
#--------------------------------参数设定-------------------------------

def read_image(path):
    '''
    input: 图片路径
    output: 图片矩阵的三个通道, 图片矩阵形状
    '''
    img = Image.open(path)
    img_array = np.asarray(img)

    red_channel = img_array[:,:,0].flatten()
    green_channel = img_array[:,:,1].flatten()
    blue_channel = img_array[:,:,2].flatten()

    return red_channel, green_channel, blue_channel, img_array.shape

def polynomial(img, n, r):
    '''
    input:
        img: 输入图像，它应该是一个2D numpy数组。
        n: 想生成的图像的数量。
        r: 多项式的阶数。
    output:
        一个形状为 (n, num_pixels) 的numpy数组，其中每一行代表一个生成的图像
    '''
    num_pixels = img.shape[0]
    coef = np.random.randint(low = 0, high = mod, size = (num_pixels, r - 1))
    gen_imgs = []
    for i in range(1, n + 1):#子秘密
        base = np.array([i ** j for j in range(1, r)])
        base = np.matmul(coef, base)
        img_ = img + base
        img_ = img_ % mod
        gen_imgs.append(img_)

    return np.array(gen_imgs)

def lagrange(x, y, num_points, x_test):
    l = np.zeros(shape=(num_points, ))

    for k in range(num_points):
        l[k] = 1
        for k_ in range(num_points):
            if k != k_:
                num = (x_test - x[k_]) % mod
                den = (x[k] - x[k_]) % mod
                inverse_den = pow(int(den), mod-2, mod)
                l[k] = (l[k] * num * inverse_den) % mod

    L = 0
    for i in range(num_points):
        L = (L + y[i]*l[i]) % mod
    return L


def decode(imgs, index, r, n):
    assert imgs.shape[0] >= r
    x = np.array(index)
    dim = imgs.shape[1]
    img = []
    for i in range(dim):
        y = imgs[:, i]
        pixel = lagrange(x, y, r, 0) % mod
        img.append(pixel)
    return np.array(img)


if __name__ == "__main__":
#------------------------------------加 密------------------------------------

    print("**************")
    print("Encrypt begin.")
    print("**************")
    r_channel, g_channel, b_channel, shape = read_image(path)

    gen_imgs_r = polynomial(r_channel, n = n, r = r)
    gen_imgs_g = polynomial(g_channel, n = n, r = r)
    gen_imgs_b = polynomial(b_channel, n = n, r = r)

    for i in range(n):
        # 将三个通道合并，得到其中一个子秘密
        combined_img = np.stack((
            gen_imgs_r[i].reshape(shape[0], shape[1]),
            gen_imgs_g[i].reshape(shape[0], shape[1]),
            gen_imgs_b[i].reshape(shape[0], shape[1])
        ), axis=-1)
        Image.fromarray(combined_img.astype(np.uint8)).save(f"test{testNum}_{i+1}.jpeg")
        print(f"test{testNum}_{i+1}.jpeg is saved.")

#------------------------------------加 密------------------------------------


#------------------------------------解 密------------------------------------
    print("**************")
    print("Decrypt begin.")
    print("**************")

    # 遍历所有图片组合以验证解密效果
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # 选择子秘密和对应的x值
                selected_subsecrets = [i, j,k]
                selected_x_values = [i+1, j+1,k+1]
                # 解密
                origin_img_r = decode(gen_imgs_r[selected_subsecrets, :], selected_x_values, r=r, n=n)
                origin_img_g = decode(gen_imgs_g[selected_subsecrets, :], selected_x_values, r=r, n=n)
                origin_img_b = decode(gen_imgs_b[selected_subsecrets, :], selected_x_values, r=r, n=n)

                # 将解密结果合并
                combined_origin_img = np.stack((
                    origin_img_r.reshape(shape[0], shape[1]),
                    origin_img_g.reshape(shape[0], shape[1]),
                    origin_img_b.reshape(shape[0], shape[1])
                ), axis=-1)

                # 保存图像，名称包含所选的子秘密的编号
                filename = f"test{testNum}_recovered_{i+1}_{j+1}_{k+1}.jpeg"
                Image.fromarray(combined_origin_img.astype(np.uint8)).save(filename)
                print(f"{filename} is saved.")


    # 挑选制定的子秘密进行解密，如下标第1,2,4张，对应x值为2,3,5
    # selected_subsecrets = [1,2,4]
    # selected_x_values = [2,3,5]

    # origin_img_r = decode(gen_imgs_r[selected_subsecrets, :], selected_x_values, r=r, n=n)
    # origin_img_g = decode(gen_imgs_g[selected_subsecrets, :], selected_x_values, r=r, n=n)
    # origin_img_b = decode(gen_imgs_b[selected_subsecrets, :], selected_x_values, r=r, n=n)


    # 将解密结果合并
    # combined_origin_img = np.stack((
    #     origin_img_r.reshape(shape[0], shape[1]),
    #     origin_img_g.reshape(shape[0], shape[1]),
    #     origin_img_b.reshape(shape[0], shape[1])
    # ), axis=-1)

    # Image.fromarray(combined_origin_img.astype(np.uint8)).save(f"test{testNum}_origin.jpeg")

    #------------------------------------解 密------------------------------------