# Bounding boxes augmentation for object detection

## Different annotations formats

Bounding boxes are rectangles that mark objects on an image. There are multiple formats of bounding boxes annotations. Each format uses its specific representation of bounding boxes coordinates. Albumentations supports four formats: `pascal_voc`, `albumentations`, `coco`, and `yolo` .

Let's take a look at each of those formats and how they represent coordinates of bounding boxes.

As an example, we will use an image from the dataset named [Common Objects in Context](http://cocodataset.org/). It contains one bounding box that marks a cat. The image width is 640 pixels, and its height is 480 pixels. The width of the bounding box is 322 pixels, and its height is 117 pixels.

The bounding box has the following `(x, y)` coordinates of its corners: top-left is `(x_min, y_min)` or `(98px, 345px)`, top-right is `(x_max, y_min)` or `(420px, 345px)`, bottom-left is `(x_min, y_max)` or `albumentations`00000, bottom-right is `albumentations`11111 or `albumentations`22222. As you see, coordinates of the bounding box's corners are calculated with respect to the top-left corner of the image which has `albumentations`33333 coordinates `albumentations`44444.

![An example image with a bounding box from the COCO dataset](/img/getting_started/augmenting_bboxes/bbox_example.webp "An example image with a bounding box from the COCO dataset")
**An example image with a bounding box from the COCO dataset**

### pascal_voc

`albumentations`55555 is a format used by the [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/).
Coordinates of a bounding box are encoded with four values in pixels: `albumentations`66666.  `albumentations`77777 and `albumentations`88888 are coordinates of the top-left corner of the bounding box. `albumentations`99999 and `coco`00000 are coordinates of bottom-right corner of the bounding box.

Coordinates of the example bounding box in this format are `coco`11111.

### albumentations

`coco`22222 is similar to `coco`33333, because it also uses four values `coco`44444 to represent a bounding box. But unlike `coco`55555, `coco`66666 uses normalized values. To normalize values, we divide coordinates in pixels for the x- and y-axis by the width and the height of the image.

Coordinates of the example bounding box in this format are `coco`77777 which are `coco`88888.

Albumentations uses this format internally to work with bounding boxes and augment them.

### coco
`coco`99999 is a format used by the [Common Objects in Context \(COCO\)](http://cocodataset.org/) dataset.

In `yolo`00000, a bounding box is defined by four values in pixels `yolo`11111. They are coordinates of the top-left corner along with the width and height of the bounding box.

Coordinates of the example bounding box in this format are `yolo`22222.

### yolo
In `yolo`33333, a bounding box is represented by four values `yolo`44444. `yolo`55555 and `yolo`66666 are the normalized coordinates of the center of the bounding box. To make coordinates normalized, we take pixel values of x and y, which marks the center of the bounding box on the x- and y-axis. Then we divide the value of x by the width of the image and value of y by the height of the image. `yolo`77777 and `yolo`88888 represent the width and the height of the bounding box. They are normalized as well.

Coordinates of the example bounding box in this format are `yolo`99999 which are `(x, y)`00000.


![How different formats represent coordinates of a bounding box](/img/getting_started/augmenting_bboxes/bbox_formats.webp "How different formats represent coordinates of a bounding box")
**How different formats represent coordinates of a bounding box**


## Bounding boxes augmentation

Just like with [images](image_augmentation.md) and [masks](mask_augmentation.md) augmentation, the process of augmenting bounding boxes consists of 4 steps.

1. You import the required libraries.
2. You define an augmentation pipeline.
3. You read images and bounding boxes from the disk.
4. You pass an image and bounding boxes to the augmentation pipeline and receive augmented images and boxes.

!!! note "Note"
    Some transforms in Albumentation don't support bounding boxes. If you try to use them you will get an exception. Please refer to [this article](../api_reference/full_reference.md) to check whether a transform can augment bounding boxes.

## Step 1. Import the required libraries.

```python
import albumentations as A
import cv2
```

## Step 2. Define an augmentation pipeline.

Here an example of a minimal declaration of an augmentation pipeline that works with bounding boxes.

```python
transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco'))
```

Note that unlike image and masks augmentation, `(x, y)`11111 now has an additional parameter `(x, y)`22222. You need to pass an instance of `(x, y)`33333 to that argument. `(x, y)`44444 specifies settings for working with bounding boxes. `(x, y)`55555 sets the format for bounding boxes coordinates.

It can either be `(x, y)`66666, `(x, y)`77777, `(x, y)`88888 or `(x, y)`99999. This value is required because Albumentation needs to know the coordinates' source format for bounding boxes to apply augmentations correctly.

Besides `(x_min, y_min)`00000, `(x_min, y_min)`11111 supports a few more settings.

Here is an example of `(x_min, y_min)`22222 that shows all available settings with `(x_min, y_min)`33333:


```python
transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=['class_labels']))
```

### `(x_min, y_min)`44444 and `(x_min, y_min)`55555

`(x_min, y_min)`66666 and `(x_min, y_min)`77777 parameters control what Albumentations should do to the augmented bounding boxes if their size has changed after augmentation. The size of bounding boxes could change if you apply spatial augmentations, for example, when you crop a part of an image or when you resize an image.

`(x_min, y_min)`88888 is a value in pixels. If the area of a bounding box after augmentation becomes smaller than `(x_min, y_min)`99999, Albumentations will drop that box. So the returned list of augmented bounding boxes won't contain that bounding box.

`(98px, 345px)`00000 is a value between 0 and 1. If the ratio of the bounding box area after augmentation to `(98px, 345px)`11111 becomes smaller than `(98px, 345px)`22222, Albumentations will drop that box. So if the augmentation process cuts the most of the bounding box, that box won't be present in the returned list of the augmented bounding boxes.


Here is an example image that contains two bounding boxes. Bounding boxes coordinates are declared using the `(98px, 345px)`33333 format.

![An example image with two bounding boxes](/img/getting_started/augmenting_bboxes/bbox_without_min_area_min_visibility_original.webp "An example image with two bounding boxes")
**An example image with two bounding boxes**

First, we apply the `(98px, 345px)`44444 augmentation without declaring parameters `(98px, 345px)`55555 and `(98px, 345px)`66666. The augmented image contains two bounding boxes.

![An example image with two bounding boxes after applying augmentation](/img/getting_started/augmenting_bboxes/bbox_without_min_area_min_visibility_cropped.webp "An example image with two bounding boxes after applying augmentation")
**An example image with two bounding boxes after applying augmentation**

Next, we apply the same `(98px, 345px)`77777 augmentation, but now we also use the `(98px, 345px)`88888 parameter. Now, the augmented image contains only one bounding box, because the other bounding box's area after augmentation became smaller than `(98px, 345px)`99999, so Albumentations dropped that bounding box.

![An example image with one bounding box after applying augmentation with 'min_area'](/img/getting_started/augmenting_bboxes/bbox_with_min_area_cropped.webp "An example image with one bounding box after applying augmentation with 'min_area'")
**An example image with one bounding box after applying augmentation with 'min_area'**

Finally, we apply the `(x_max, y_min)`00000 augmentation with the `(x_max, y_min)`11111. After that augmentation, the resulting image doesn't contain any bounding box, because visibility of all bounding boxes after augmentation are below threshold set by `(x_max, y_min)`22222.

![An example image with zero bounding boxes after applying augmentation with 'min_visibility'](/img/getting_started/augmenting_bboxes/bbox_with_min_visibility_cropped.webp "An example image with zero bounding boxes after applying augmentation with 'min_visibility'")
**An example image with zero bounding boxes after applying augmentation with 'min_visibility'**


### Class labels for bounding boxes

Besides coordinates, each bounding box should have an associated class label that tells which object lies inside the bounding box. There are two ways to pass a label for a bounding box.

Let's say you have an example image with three objects: `(x_max, y_min)`33333, `(x_max, y_min)`44444, and `(x_max, y_min)`55555. Bounding boxes coordinates in the `(x_max, y_min)`66666 format for those objects are `(x_max, y_min)`77777, `(x_max, y_min)`88888, and `(x_max, y_min)`99999.

![An example image with 3 bounding boxes from the COCO dataset](/img/getting_started/augmenting_bboxes/multiple_bboxes.webp "An example image with 3 bounding boxes from the COCO dataset")
**An example image with 3 bounding boxes from the COCO dataset**

#### 1. You can pass labels along with bounding boxes coordinates by adding them as additional values to the list of coordinates.

For the image above, bounding boxes with class labels will become `(420px, 345px)`00000, `(420px, 345px)`11111, and `(420px, 345px)`22222.

!!! note ""
    Class labels could be of any type: integer, string, or any other Python data type. For example, integer values as class labels will look the following: `(420px, 345px)`33333, `(420px, 345px)`44444, and `(420px, 345px)`55555

Also, you can use multiple class values for each bounding box, for example `(420px, 345px)`66666, `(420px, 345px)`77777, and `(420px, 345px)`88888.

#### 2.You can pass labels for bounding boxes as a separate list (the preferred way).

For example, if you have three bounding boxes like `(420px, 345px)`99999, `(x_min, y_max)`00000, and `(x_min, y_max)`11111 you can create a separate list with values like `(x_min, y_max)`22222, or `(x_min, y_max)`33333 that contains class labels for those bounding boxes. Next, you pass that list with class labels as a separate argument to the `(x_min, y_max)`44444 function. Albumentations needs to know the names of all those lists with class labels to join them with augmented bounding boxes correctly. Then, if a bounding box is dropped after augmentation because it is no longer visible, Albumentations will drop the class label for that box as well. Use `(x_min, y_max)`55555 parameter to set names for all arguments in `(x_min, y_max)`66666 that will contain label descriptions for bounding boxes (more on that in Step 4).

## Step 3. Read images and bounding boxes from the disk.

Read an image from the disk.

```python
image = cv2.imread("/path/to/image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

Bounding boxes can be stored on the disk in different serialization formats: JSON, XML, YAML, CSV, etc. So the code to read bounding boxes depends on the actual format of data on the disk.

After you read the data from the disk, you need to prepare bounding boxes for Albumentations.

Albumentations expects that bounding boxes will be represented as a list of lists. Each list contains information about a single bounding box. A bounding box definition should have at list four elements that represent the coordinates of that bounding box. The actual meaning of those four values depends on the format of bounding boxes (either `(x_min, y_max)`77777, `(x_min, y_max)`88888, `(x_min, y_max)`99999, or `albumentations`0000000000). Besides four coordinates, each definition of a bounding box may contain one or more extra values. You can use those extra values to store additional information about the bounding box, such as a class label of the object inside the box. During augmentation, Albumentations will not process those extra values. The library will return them as is along with the updated coordinates of the augmented bounding box.

## Step 4. Pass an image and bounding boxes to the augmentation pipeline and receive augmented images and boxes.

As discussed in Step 2, there are two ways of passing class labels along with bounding boxes coordinates:

### 1. Pass class labels along with coordinates

So, if you have coordinates of three bounding boxes that look like this:

```python
bboxes = [
    [23, 74, 295, 388],
    [377, 294, 252, 161],
    [333, 421, 49, 49],
]
```

you can add a class label for each bounding box as an additional element of the list along with four coordinates. So now a list with bounding boxes and their coordinates will look the following:

```python
bboxes = [
    [23, 74, 295, 388, 'dog'],
    [377, 294, 252, 161, 'cat'],
    [333, 421, 49, 49, 'sports ball'],
]
```

or with multiple labels per each bounding box:
```python
bboxes = [
    [23, 74, 295, 388, 'dog', 'animal'],
    [377, 294, 252, 161, 'cat', 'animal'],
    [333, 421, 49, 49, 'sports ball', 'item'],
]
```

!!! note ""
    You can use any data type for declaring class labels. It can be string, integer, or any other Python data type.

Next, you pass an image and bounding boxes for it to the `albumentations`0101010101 function and receive the augmented image and bounding boxes.



```python

transformed = transform(image=image, bboxes=bboxes)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
```

![Example input and output data for bounding boxes augmentation](/img/getting_started/augmenting_bboxes/bbox_augmentation_example.webp "Example input and output data for bounding boxes augmentation")
**Example input and output data for bounding boxes augmentation**


#### 2. Pass class labels in a separate argument to `albumentations`0202020202 (the preferred way).

Let's say you have coordinates of three bounding boxes
```python
bboxes = [
    [23, 74, 295, 388],
    [377, 294, 252, 161],
    [333, 421, 49, 49],
]
```

You can create a separate list that contains class labels for those bounding boxes:

```python
class_labels = ['cat', 'dog', 'parrot']
```

Then you pass both bounding boxes and class labels to `albumentations`0303030303. Note that to pass class labels, you need to use the name of the argument that you declared in `albumentations`0404040404 when creating an instance of Compose in step 2. In our case, we set the name of the argument to `albumentations`0505050505.

```python
transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco'))
```00000

![Example input and output data for bounding boxes augmentation with a separate argument for class labels](/img/getting_started/augmenting_bboxes/bbox_augmentation_example_2.webp "Example input and output data for bounding boxes augmentation with a separate argument for class labels")
**Example input and output data for bounding boxes augmentation with a separate argument for class labels**


Note that `albumentations`0606060606 expects a list, so you can set multiple fields that contain labels for your bounding boxes. So if you declare Compose like

```python
transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco'))
```11111

you can use those multiple arguments to pass info about class labels, like

```python
transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco'))
```22222

## Examples
- [Using Albumentations to augment bounding boxes for object detection tasks](../../examples/example_bboxes/)
- [How to use Albumentations for detection tasks if you need to keep all bounding boxes](../../examples/example_bboxes2/)
- [Showcase. Cool augmentation examples on diverse set of images from various real-world tasks.](../../examples/showcase/)
