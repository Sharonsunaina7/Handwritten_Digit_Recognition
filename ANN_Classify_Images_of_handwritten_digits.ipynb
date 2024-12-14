{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "id": "Tk9zkwXEdzMr"
   },
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential, load_model\n",
    "from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "xkFwdJ16etTB",
    "outputId": "fe5f1f41-a85e-455c-ecc1-3fde656eba6f"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\LENOVO\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\keras\\src\\layers\\convolutional\\base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
     ]
    }
   ],
   "source": [
    "# Let's build a brain\n",
    "model = Sequential()\n",
    "\n",
    "# First, we'll add a layer that looks at small parts of the picture\n",
    "model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))\n",
    "\n",
    "# Then, we'll add a layer that picks the most important parts\n",
    "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "\n",
    "# We'll add another layer to look at the important parts more closely\n",
    "model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))\n",
    "\n",
    "# And another layer to pick the most important parts again\n",
    "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "\n",
    "# Now, we'll flatten the picture into a long line\n",
    "model.add(Flatten())\n",
    "\n",
    "# We'll add a layer to think about the line and make a guess\n",
    "model.add(Dense(128, activation='relu'))\n",
    "\n",
    "# Finally, we'll make a final guess about the number\n",
    "model.add(Dense(10, activation='softmax'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "C9JhWX5jg2Sk"
   },
   "source": [
    "# Conv2D (Convolutional Layer):\n",
    "Imagine you have a picture (like a photo of a cat). Now, you want to find certain things in the picture, like ears, eyes, or whiskers. A Conv2D layer is like a little \"window\" that slides over the picture, looking for things like edges and shapes. Each time the window slides over a part of the picture, it tries to find a little part of the cat, like an ear or an eye.\n",
    "\n",
    "# MaxPooling2D (Pooling Layer):\n",
    "After the Conv2D layer finds important parts like ears or eyes, we don’t need all the little details anymore. So, the MaxPooling2D layer helps by shrinking the picture a little and keeping only the most important parts, like the biggest shapes. It’s like looking at a picture through a small window and saying, \"What’s the biggest thing in this part?\" and remembering that."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "id": "plMUIZ5RezyE"
   },
   "outputs": [],
   "source": [
    "model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])\n",
    "\n",
    "(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()\n",
    "\n",
    "x_train = x_train / 255.0\n",
    "x_test = x_test / 255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "e7NSs9tLe2js",
    "outputId": "1d3b1151-7388-46f7-d133-9cb18160182b"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m19s\u001b[0m 9ms/step - accuracy: 0.9107 - loss: 0.2902\n",
      "Epoch 2/10\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m19s\u001b[0m 10ms/step - accuracy: 0.9862 - loss: 0.0466\n",
      "Epoch 3/10\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m18s\u001b[0m 9ms/step - accuracy: 0.9904 - loss: 0.0298\n",
      "Epoch 4/10\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m17s\u001b[0m 9ms/step - accuracy: 0.9935 - loss: 0.0199\n",
      "Epoch 5/10\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m18s\u001b[0m 10ms/step - accuracy: 0.9955 - loss: 0.0139\n",
      "Epoch 6/10\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m17s\u001b[0m 9ms/step - accuracy: 0.9964 - loss: 0.0101\n",
      "Epoch 7/10\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m17s\u001b[0m 9ms/step - accuracy: 0.9967 - loss: 0.0097\n",
      "Epoch 8/10\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m17s\u001b[0m 9ms/step - accuracy: 0.9970 - loss: 0.0075\n",
      "Epoch 9/10\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m17s\u001b[0m 9ms/step - accuracy: 0.9980 - loss: 0.0061\n",
      "Epoch 10/10\n",
      "\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m17s\u001b[0m 9ms/step - accuracy: 0.9974 - loss: 0.0081\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.history.History at 0x1d69c0b2b90>"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x_train, y_train, epochs=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "id": "vJkGtX7Cd10X"
   },
   "outputs": [],
   "source": [
    "img_path = r\"C:\\Users\\LENOVO\\Desktop\\Skilligence AI-ML\\new task\\DataSets\\DataSets\\5.jpg\"\n",
    "img = tf.keras.utils.load_img(img_path, color_mode=\"grayscale\", target_size=(28, 28))\n",
    "img_array = tf.keras.utils.img_to_array(img)\n",
    "# img_array = 255 - img_array  # Invert colors if needed\n",
    "img_array = np.expand_dims(img_array, axis=0)\n",
    "img_array /= 255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "zM7MFZ9zeQ_z",
    "outputId": "60acf7d0-9c2b-46be-8fbb-e6591db2e5a4"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 114ms/step\n"
     ]
    }
   ],
   "source": [
    "predictions = model.predict(img_array)\n",
    "predicted_digit = np.argmax(predictions[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 447
    },
    "id": "ns9oHHeKfCU3",
    "outputId": "44acbf62-a076-49bb-a726-f4ca4c19733b"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted digit: 2\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAaAAAAGdCAYAAABU0qcqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy80BEi2AAAACXBIWXMAAA9hAAAPYQGoP6dpAAAYdUlEQVR4nO3df0zU9x3H8depcNUWjiLCcRUpaqtJrWxzyoirayJR3GLqjz9c1z/sYmy0ZzN17RaXKHZZwmaTZuli1v2lWVq1Mxma+oeJomC2oU2txph1RAgbGDlcTfgeoqCBz/6g3noK6OGd7wOfj+SdyN0XePvdleeO+wo+55wTAACP2DjrBQAAjycCBAAwQYAAACYIEADABAECAJggQAAAEwQIAGCCAAEATEywXuBu/f39unLlirKysuTz+azXAQAkyDmnrq4uhUIhjRs39POctAvQlStXVFRUZL0GAOAhtbW1aerUqUPen3YBysrKsl4BAPANnucldHw0GlVRUdF9v56n7DWg3bt369lnn9UTTzyhsrIyffbZZw/0fnzbDQDSS3Z2dsIj3f/reUoC9Mknn2jr1q2qqqrSF198odLSUi1dulRXr15NxacDAIxGLgUWLFjgwuFw7O2+vj4XCoVcdXX1fd/X8zwniWEYhkmTSdSdr+Oe5w17XNKfAd26dUtnz55VRUVF7LZx48apoqJCDQ0N9xzf29uraDQaNwCAsS/pAfrqq6/U19engoKCuNsLCgoUiUTuOb66ulqBQCA2XAEHAI8H83+Ium3bNnmeF5u2tjbrlQAAj0DSL8POy8vT+PHj1dHREXd7R0eHgsHgPcf7/X75/f5krwEASHNJfwaUmZmpefPmqba2NnZbf3+/amtrVV5enuxPBwAYpVLyD1G3bt2qtWvX6rvf/a4WLFig3//+9+ru7tZPf/rTVHw6AMAolJIArVmzRv/973+1Y8cORSIRfetb39LRo0fvuTABAPD48jnnnPUS3xSNRhUIBKzXAAB8LdFM3Pk67nle7KciDMb8KjgAwOOJAAEATBAgAIAJAgQAMEGAAAAmCBAAwAQBAgCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAEwQIAGCCAAEATEywXuBx4ZyzXmHU8vl81isASAGeAQEATBAgAIAJAgQAMEGAAAAmCBAAwAQBAgCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJvhhpI8IP1ATAOLxDAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAEwQIAGCCAAEATBAgAIAJAgQAMEGAAAAmCBAAwAQBAgCYSHqAdu7cKZ/PFzezZ89O9qcBAIxyKfmFdC+88IKOHz/+/08ygd97BwCIl5IyTJgwQcFgMBUfGgAwRqTkNaBLly4pFApp+vTpeu2119Ta2jrksb29vYpGo3EDABj7kh6gsrIy7d27V0ePHtUf//hHtbS06KWXXlJXV9egx1dXVysQCMSmqKgo2SsBANKQzznnUvkJOjs7VVxcrPfff1/r1q275/7e3l719vbG3o5Go0QIANJIopmIRqMKBALyPE/Z2dlDHpfyqwNycnL0/PPPq6mpadD7/X6//H5/qtcAAKSZlP87oOvXr6u5uVmFhYWp/lQAgFEk6QF6++23VV9fr3//+9/6xz/+oZUrV2r8+PF69dVXk/2pAACjWNK/BXf58mW9+uqrunbtmqZMmaLvf//7On36tKZMmZLsTwUAGMWSHqADBw4k+0MCY85Irv3x+Xwp2ASww8+CAwCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAEwQIAGCCAAEATBAgAIAJAgQAMEGAAAAmCBAAwAQBAgCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYmWC8API58Pp/1CoA5ngEBAEwQIACACQIEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAEwkH6NSpU1q+fLlCoZB8Pp8OHToUd79zTjt27FBhYaEmTpyoiooKXbp0KVn7AgDGiIQD1N3drdLSUu3evXvQ+3ft2qUPPvhAH374oc6cOaMnn3xSS5cuVU9Pz0MvCwAYQ9xDkORqampib/f397tgMOjee++92G2dnZ3O7/e7/fv3P9DH9DzPSWIYhmHSZBJ1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r69PBQUFcbcXFBToX//6l9FWNsrKyrR3717NmjVL7e3tevfdd/XSSy/p4sWLysrKsl7PRCQSkaRBHx937ntcVFZWatWqVSopKVFzc7N+9atfadmyZWpoaND48eOt10u6/v5+bd68WQsXLtScOXMkDTweMjMzlZOTE3fsWH48DHYeJOknP/mJiouLFQqFdOHCBf3yl79UY2Oj/vrXvxpuGy/tA4T/W7ZsWezPc+fOVVlZmYqLi/WXv/xF69atM9wM6eDHP/5x7M8vvvii5s6dqxkzZqiurk6LFy823Cw1wuGwLl68+Fi8Djqcoc7DG2+8Efvziy++qMLCQi1evFjNzc2aMWPGo15zUGn/Lbi8vDyNHz/+nqtYOjo6FAwGjbZKDzk5OXr++efV1NRkvYqZO48BHh/3mj59uvLy8sbk42PTpk06cuSITp48GffrW4LBoG7duqXOzs6448fq42Go8zCYsrIySUqrx0PaBygzM1Pz5s1TbW1t7Lb+/n7V1taqvLzccDN7169fV3NzswoLC61XMVNSUqJgMBj3+IhGozpz5sxj//i4fPmyrl27NqYeH845bdq0STU1NTpx4oRKSkri7p83b54yMjLiHg+NjY1qbW0dU4+H+52HwZw/f16S0uvxYH0VxIM4cOCA8/v9bu/eve6f//yne+ONN1xOTo6LRCLWqz1SP//5z11dXZ1raWlxf//7311FRYXLy8tzV69etV4tpbq6uty5c+fcuXPnnCT3/vvvu3Pnzrn//Oc/zjnnfvvb37qcnBx3+PBhd+HCBffKK6+4kpISd/PmTePNk2u489DV1eXefvtt19DQ4FpaWtzx48fdd77zHffcc8+5np4e69WTZuPGjS4QCLi6ujrX3t4emxs3bsSO2bBhg5s2bZo7ceKE+/zzz115ebkrLy833Dr57ncempqa3K9//Wv3+eefu5aWFnf48GE3ffp0t2jRIuPN442KADnn3B/+8Ac3bdo0l5mZ6RYsWOBOnz5tvdIjt2bNGldYWOgyMzPdM88849asWeOampqs10q5kydPOkn3zNq1a51zA5dib9++3RUUFDi/3+8WL17sGhsbbZdOgeHOw40bN9ySJUvclClTXEZGhisuLnbr168fc/8nbbC/vyS3Z8+e2DE3b950b775pnv66afdpEmT3MqVK117e7vd0ilwv/PQ2trqFi1a5HJzc53f73czZ85077zzjvM8z3bxu/DrGAAAJtL+NSAAwNhEgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJj4H6IxkwCtzUPtAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"Predicted digit:\", predicted_digit)\n",
    "plt.imshow(img_array[0], cmap='gray')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "5MkWL6bogT01"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
