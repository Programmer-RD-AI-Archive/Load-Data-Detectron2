def load_data( test: bool = False) -> list:
    """
    - load_data - loading the data in the detectron2 data format
    -------------------------------------
    - test - if the return is supposed to be a test sample or not
        - Defalt = False and type = bool
    """
    if test is True:
        if "data.npy" in os.listdir("./"):
            data = np.load(
                "./data.npy", allow_pickle=True
            )  # Loading already saved detectron2 format file
            data = data[: test_sample_size]  # TODO
            return data
    if "data.npy" in os.listdir("./"):
        data = np.load("./data.npy", allow_pickle=True)
        return data
    new_data = []
    for idx in tqdm(range(len(data))):  # iter over the data
        record = {}
        info = data.iloc[idx]
        height, width = cv2.imread("./Img/" + info["Path"]).shape[:2]
        xmin, ymin, xmax, ymax = (
            info["XMin"],
            info["YMin"],
            info["XMax"],
            info["YMax"],
        )
        xmin = round(xmin * width)
        xmax = round(xmax * width)
        ymin = round(ymin * height)
        ymax = round(ymax * height)
        record["file_name"] = "./Img/" + info["Path"]
        record["height"] = height
        record["width"] = width
        objs = [
            {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
            }
        ]
        record["image_id"] = idx
        record["annotations"] = objs
        new_data.append(record)
    np.random.shuffle(new_data)  # Shuffling the data
    # np.save("data.npy", new_data)  # Saving the data
    if test is True:
        return new_data[: test_sample_size]
    return new_data
