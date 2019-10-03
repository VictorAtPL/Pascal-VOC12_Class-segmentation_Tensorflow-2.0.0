from matplotlib import pyplot as plt
from VOClabelcolormap import color_map
from common import load_sets_count, get_input_fn_and_steps_per_epoch, \
    blend_from_3d_sparse, pascal_voc2012_segmentation_annotated_parser
from constants import TFRECORDS_SAVE_PATH


def main():
    cmap = color_map()

    sets_count = load_sets_count()

    parser_fn = pascal_voc2012_segmentation_annotated_parser

    input_fn, _ = get_input_fn_and_steps_per_epoch('train', parser_fn, TFRECORDS_SAVE_PATH, 4, sets_count)

    it = iter(input_fn)

    images, annotations = next(it)

    fig = plt.figure()
    image = images.numpy()[0]
    annotation = annotations.numpy()[0]

    blended = blend_from_3d_sparse(image, annotation, cmap, alpha=0.5)
    plt.imshow(blended)
    fig.show()


if __name__ == '__main__':
    main()
