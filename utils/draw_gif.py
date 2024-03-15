import imageio
import os


# import torchvision as tv


def create_gif(image_list, gif_name, duration=1.0):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))

    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def main():
    # image_list = ['1.jpg', '2.jpg', '3.jpg']
    for j in range(1, 11):
        image_list = [f"./figs/figs/{j}_3_4_{i}.png" for i in range(1, 172)]
        image_list.append('./white.png')
        gif_name = f"{j}_cmp_before.gif"
        duration = 1.5
        create_gif(image_list, gif_name, duration)

        image_list = [f"./figs/figs/{j}_3_4_{i}.png" for i in range(171, 485)]
        image_list.append('./white.png')
        gif_name = f"{j}_cmp_after.gif"
        duration = 0.2
        create_gif(image_list, gif_name, duration)
        print(f"finishing {j} th")


if __name__ == '__main__':
    pass
