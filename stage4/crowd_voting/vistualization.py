import cv2
def show(net, patiend_id):
    files = np.loadtxt("testfile_name.txt", dtype=np.str, delimiter='/n')
    with open(crowd_vote_result, 'r') as reader:
      reader = reader.read().splitlines()
      for line in reader:
        vote_result_single = line.split(",")
        box = [float(vote_result_single[2]), float(vote_result_single[1]), 
              float(vote_result_single[4]),float(vote_result_single[3])]
        image_dir = vote_result_single[0][:30]
        evaluate_single_file(box, image_dir)
    for i in range(len(files)) {
        if (files[i].contains(patiend_id)) {
            im_fname = files[i]
            print(im_fname)
            im = cv2.imread(im_fname)
            x, img = data.transforms.presets.yolo.load_test(im_fname, short=im.shape[0])
            all_boxes = np.array(getBbox(im_fname.replace('.tif', '.csv')))
            all_ids = np.array([0])
            class_names = ['bleeding']

            # see how it looks by rendering the boxes into image
            ax = utils.viz.plot_bbox(img, all_boxes, labels=all_ids, class_names=class_names, colors={0: (0, 1, 0)})

            x, img = data.transforms.presets.yolo.load_test(im_fname, short=im.shape[0])

        }
    }
    print(im_fname)
    im = cv2.imread(im_fname)
    print(im.shape)
    x, img = data.transforms.presets.yolo.load_test(im_fname, short=im.shape[0])
    print('Shape of pre-processed image:', x.shape)
    class_IDs, scores, bounding_boxs = net(x)

    ax = utils.viz.plot_bbox(img, bounding_boxs[0][:first_n_prediction], scores[0][:first_n_prediction],
                            class_IDs[0][:first_n_prediction], class_names=net.classes, thresh=0)
    plt.show()

    x, img = data.transforms.presets.yolo.load_test(im_fname, short=im.shape[0])
    all_boxes = np.array(getBbox(im_fname.replace('.tif', '.csv')))
    all_ids = np.array([0])
    class_names = ['bleeding']

    # see how it looks by rendering the boxes into image
    ax = utils.viz.plot_bbox(img, all_boxes, labels=all_ids, class_names=class_names, colors={0: (0, 1, 0)})
    plt.show()